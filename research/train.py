"""Strict image-label-only full-dataset training and auditable full-val evaluation."""
from pathlib import Path
import argparse,hashlib,json,os,random,shutil,subprocess,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from research.model_factory import build_model
from research.protocol import validate_full_splits,spec,mask_path
from research.data import ImageLabelDataset,load_eval_image,load_eval_mask
from research.confusions import ConfusionStore
from research.methods import anchored_graph,partial_label_loss
from utils.affutils import refine_cams_with_aff,refine_cams_with_bkg_weclip
from utils.PAR import PAR
from model.losses import get_seg_loss,get_aff_loss
from model.dice_loss import dice_loss
from research.release_version import source_revision
from utils.camutils import cams_to_affinity_label,get_mask_by_radius


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()

def write_json(path, data):
    path=Path(path); tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(path)

def append(path,data):
    with open(path,'a',encoding='utf-8') as f: f.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)

def metric(hist):
    denom=hist.sum(0)+hist.sum(1)-hist.diagonal()
    iou=np.divide(hist.diagonal(),denom,out=np.zeros(hist.shape[0],dtype=float),where=denom>0)
    assert (hist.sum(1)>0).all(), 'All configured classes must be represented in full validation'
    return {'miou_percent':float(iou.mean()*100),'iou_percent':(iou*100).tolist(),'confusion':hist.tolist()}

@torch.no_grad()
def evaluate(model,cfg,folder,step,scales=(1.,),flip=False,limit=0):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    _,ids=validate_full_splits(cfg)
    nc=spec(cfg)['classes']
    if limit: ids=ids[:limit]
    tag=f'eval_{step}_'+('multiscale' if len(scales)>1 else 'single')
    individual=ConfusionStore(folder/f'{tag}_confusions.npz',len(ids),nc)
    hist=np.zeros((nc,nc),np.int64); started=time.time()
    model.eval()
    for j,name in enumerate(ids):
        raw,x=load_eval_image(cfg['data'],name,cfg); x=x.cuda()
        label=load_eval_mask(cfg,name)
        probs=[]
        for scale in scales:
            size=max(32,int(round(cfg['crop']*scale/16))*16)
            inp=F.interpolate(x,(size,size),mode='bilinear',align_corners=False)
            logits=model(inp)['seg']
            if flip: logits=(logits+model(inp.flip(-1))['seg'].flip(-1))*.5
            probs.append(F.interpolate(logits.float(),label.shape,mode='bilinear',align_corners=False).softmax(1))
        pred=torch.stack(probs).mean(0).argmax(1)[0].cpu().numpy()
        valid=(label>=0)&(label<nc)
        mat=np.bincount(nc*label[valid].astype(np.int64)+pred[valid],minlength=nc*nc).reshape(nc,nc)
        hist+=mat; individual.append(mat)
        if (j+1)%200==0: print(f'EVAL step={step} images={j+1}/{len(ids)} seconds={time.time()-started:.1f}',flush=True)
    if limit:
        denom=hist.sum(0)+hist.sum(1)-hist.diagonal()
        iou=np.divide(hist.diagonal(),denom,out=np.zeros(nc),where=denom>0)
        result={'smoke_only':True,'miou_percent':None,'class_iou_observed':iou.tolist()}
    else: result=metric(hist)
    result.update({'dataset':cfg['dataset'],'num_classes':nc,'train_n':spec(cfg)['train'],'train_fraction':1.0,
                   'step':step,'n':len(ids),'scales':list(scales),'flip':flip,'crf':False,
                   'image_label_masking':False,'val_sha256':sha(cfg['val_ids']),'wall_seconds':time.time()-started})
    individual.export(ids)
    write_json(folder/f'{tag}.json',result)
    model.train()
    return result

def checkpoint(model,optimizer,cfg,step,out):
    # Frozen encoders are reconstructed from strictly verified source weights.
    state={k:v.cpu() for k,v in model.state_dict().items() if not k.startswith(('encoder.','dense.net.'))}
    payload={'model':state,'config':cfg,'step':step,'optimizer':optimizer.state_dict(),
             'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()}
    temp=Path(str(out)+'.tmp'); torch.save(payload,temp); temp.replace(out)

def load_heads(model,path):
    p=torch.load(path,map_location='cpu',weights_only=True)
    result=model.load_state_dict(p['model'],strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith(('encoder.','dense.net.')) for k in result.missing_keys),result
    return p

def run(cfg,run_dir,smoke=False,eval_checkpoint=None):
    train_ids,val_ids=validate_full_splits(cfg)
    nc=spec(cfg)['classes']
    if not smoke and cfg['iters'] < (len(train_ids)+cfg['batch']-1)//cfg['batch']:
        raise ValueError('Training budget must cover at least one complete pass through all images')
    candidate_steps=sorted({max(1,int(cfg['iters']*.9)),cfg['iters']})
    cfg = dict(cfg, evaluation_policy='post_training_selection', eval_every=cfg['iters'], candidate_steps=candidate_steps)
    run_dir=Path(run_dir); run_dir.mkdir(parents=True,exist_ok=True)
    seed=cfg['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    os.environ['CLIP_CACHE_DIR']=cfg['clip_cache']
    model=build_model(cfg).cuda().train()
    if eval_checkpoint:
        payload=load_heads(model,eval_checkpoint)
        result=evaluate(model,cfg,run_dir,payload['step'],scales=cfg.get('eval_scales',[1.]),flip=cfg.get('eval_flip',False))
        result['checkpoint_sha256']=sha(eval_checkpoint); write_json(run_dir/'final_eval.json',result); print(json.dumps(result)); return
    params=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    par=PAR(num_iter=20,dilations=[1,2,4,8,12,24]).cuda()
    ds=ImageLabelDataset(cfg['data'],cfg['train_ids'],cfg['cls_labels'],cfg['crop'],
                         cfg['dataset'],nc,cfg)
    assert ds.ids==train_ids
    loader=DataLoader(ds,batch_size=cfg['batch'],num_workers=cfg['workers'],shuffle=True,drop_last=False,
                      worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(seed),pin_memory=True)
    it=iter(loader); seen=set(); start=time.time(); best=-1.; count=20 if smoke else cfg['iters']; candidates=[]
    commit=source_revision()
    provenance={'config':cfg,'code_commit':commit,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                'train_sha256':sha(cfg['train_ids']),'val_sha256':sha(cfg['val_ids']),
                'dino_sha256':sha(cfg['dino_weights']),'clip_sha256':sha(Path(cfg['clip_cache'])/'ViT-B-16.pt'),
                'dataset':cfg['dataset'],'num_classes':nc,'train_fraction':1.0,'train_n':len(ds),'val_n':len(val_ids),'dense_gt_read_during_training':False,
                'initialization':'generic pretrained encoders, freshly initialized decoder; no dataset teacher checkpoint',
                'trainable_parameters':sum(p.numel() for p in params),'argv':sys.argv}
    write_json(run_dir/'provenance.json',provenance)
    write_json(run_dir/'config.json',cfg)
    (run_dir/'train_ids.txt').write_text('\n'.join(ds.ids)+'\n')
    (run_dir/'val_ids.txt').write_text('\n'.join(val_ids)+'\n')
    radius_mask=get_mask_by_radius(h=cfg['crop']//16,w=cfg['crop']//16,radius=8)
    for step in range(1,count+1):
        try: names,inputs,tags,boxes=next(it)
        except StopIteration: it=iter(loader); names,inputs,tags,boxes=next(it)
        seen.update(names); inputs=inputs.cuda(non_blocking=True); tags=tags.cuda(non_blocking=True)
        out=model(inputs)
        h,w=inputs.shape[-2:]; pseudo=[]; maps=[]
        mean=inputs.new_tensor([.485,.456,.406])[None,:,None,None]
        std=inputs.new_tensor([.229,.224,.225])[None,:,None,None]
        raw=(inputs*std+mean).clamp(0,1)
        with torch.no_grad():
            for i,cam in enumerate(out['cams']):
                seg_aff=out['affinity'][i:i+1].detach() if step>=cfg.get('learned_affinity_start',7000) else None
                refined,classes=refine_cams_with_aff(cam,out['attention'][:,i],tags[i],(h,w),caa_thre=.79,seg_attn=seg_aff)
                densemap=inputs.new_zeros((nc-1,h//16,w//16))
                for j,k in enumerate(classes): densemap[k]=F.interpolate(refined[j][None,None].float(),(h//16,w//16),mode='bilinear',align_corners=False)[0,0]
                maps.append(densemap)
            maps=torch.stack(maps).clamp_min(0)
            maps=maps/maps.flatten(2).amax(-1)[:,:,None,None].clamp_min(1e-6)
            prototype=None
            if cfg['graph'] or cfg['partial']:
                enhanced,prototype=anchored_graph(maps,out['dense'],tags,topk=cfg.get('topk',16),steps=cfg.get('graph_steps',3),strength=cfg.get('graph_strength',.5))
                if cfg['graph']: maps=enhanced
            for i in range(len(inputs)):
                classes=tags[i].nonzero().flatten()
                target,_=refine_cams_with_bkg_weclip(list(maps[i,classes]),raw[i],classes.cpu(),par,(h,w))
                # Mask padding using image geometry; no GT segmentation loaded.
                box=boxes[i].tolist(); mask=torch.zeros_like(target,dtype=torch.bool)
                mask[:,max(0,box[0]):min(h,box[1]),max(0,box[2]):min(w,box[3])]=True
                target[~mask]=255; pseudo.append(target)
            pseudo=torch.cat(pseudo)
        logits=F.interpolate(out['seg'],(h,w),mode='bilinear',align_corners=False)
        conflict=logits.new_tensor(0.)
        if cfg['partial'] and step>=cfg.get('partial_start',2000):
            ce,conflict=partial_label_loss(logits,pseudo,prototype,tags)
        else: ce=get_seg_loss(logits,pseudo,use_dice=False)
        dice=dice_loss(logits,pseudo)
        targets=cams_to_affinity_label(pseudo,mask=radius_mask)
        aff,_,_=get_aff_loss(out['affinity'],targets)
        dice_weight=.1+.9*min(step/cfg['iters'],1.)
        loss=ce+dice_weight*dice+.1*aff
        if not torch.isfinite(loss): raise FloatingPointError(f'nonfinite loss at {step}')
        lr=cfg['lr']*min(step/max(1,cfg.get('warmup',100)),1.)*(1-step/(cfg['iters']+1))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,10.)
        if not torch.isfinite(grad): raise FloatingPointError(f'nonfinite gradients at {step}')
        optimizer.step()
        if step==1 or step%cfg.get('log_every',100)==0 or step==count:
            record={'step':step,'loss':loss.item(),'ce':ce.item(),'dice':dice.item(),'affinity':aff.item(),
                    'partial_fraction':conflict.item(),'ignore_fraction':(pseudo==255).float().mean().item(),
                    'lr':lr,'wall_seconds':time.time()-start,'peak_mib':torch.cuda.max_memory_allocated()/2**20,'unique_train_seen':len(seen)}
            append(run_dir/'metrics.jsonl',record); write_json(run_dir/'status.json',dict(status='training',**record))
        if not smoke and step in candidate_steps:
            path=run_dir/('last.pt' if step==count else f'candidate_{step}.pt')
            checkpoint(model,optimizer,cfg,step,path)
            candidates.append((step,path))  # Save only: never validate inside the training loop.
    (run_dir/'observed_train_ids.txt').write_text('\n'.join(sorted(seen))+'\n')
    if not smoke:
        if seen != set(train_ids):
            raise RuntimeError(f'Incomplete training coverage: {len(seen)}/{len(train_ids)}')
        write_json(run_dir/'status.json',{'status':'selecting_checkpoint_after_training','step':count})
        for candidate_step,path in candidates:
            load_heads(model,path)
            result=evaluate(model,cfg,run_dir,candidate_step)
            result.update(optimizer_steps_completed=count,selection='two_prespecified_terminal_candidates_after_training')
            write_json(run_dir/f'eval_{candidate_step}_single.json',result)
            append(run_dir/'evaluations.jsonl',result)
            if result['miou_percent']>best:
                best=result['miou_percent'];shutil.copy2(path,run_dir/'best.pt')
                write_json(run_dir/'best.json',dict(result,checkpoint_sha256=sha(run_dir/'best.pt')))
    if smoke:
        checkpoint(model,optimizer,cfg,count,run_dir/'smoke.pt')
        load_heads(model,run_dir/'smoke.pt')
        evaluate(model,cfg,run_dir,count,limit=2)
    write_json(run_dir/'status.json',{'status':'smoke_passed' if smoke else 'completed','step':count,
                 'best_miou_percent':None if smoke else best,'wall_seconds':time.time()-start,'unique_train_seen':len(seen)})

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--run-dir',required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--evaluate'); a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text())
    try: run(cfg,a.run_dir,a.smoke,a.evaluate)
    except Exception as e:
        Path(a.run_dir).mkdir(parents=True,exist_ok=True)
        write_json(Path(a.run_dir)/'status.json',{'status':'failed','error':repr(e),'time':time.time()}); raise

if __name__=='__main__': main()
