"""CAM seed evaluation. Pixel GT is used for metrics/plots, never model input."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from research.protocol import validate_full_splits,read_ids,spec,image_path,mask_path
from research.data import load_eval_mask
from research.confusions import ConfusionStore

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def confusion(gt,pred,num_classes=21):
    valid=(gt>=0)&(gt<num_classes)
    return np.bincount(num_classes*gt[valid].astype(np.int64)+pred[valid],minlength=num_classes**2).reshape(num_classes,num_classes)
def score(m):
    h=m.sum(0);tp=h.diagonal();union=h.sum(0)+h.sum(1)-tp
    iou=np.divide(tp,union,out=np.zeros(h.shape[0]),where=union>0)
    precision=np.divide(tp,h.sum(0),out=np.zeros(h.shape[0]),where=h.sum(0)>0)
    recall=np.divide(tp,h.sum(1),out=np.zeros(h.shape[0]),where=h.sum(1)>0)
    assert (h.sum(1)>0).all()
    return {'miou_percent':float(iou.mean()*100),'precision_macro_percent':float(precision.mean()*100),
            'recall_macro_percent':float(recall.mean()*100),'iou_percent':(iou*100).tolist(),
            'precision_percent':(precision*100).tolist(),'recall_percent':(recall*100).tolist(),'confusion':h.tolist()}

@torch.no_grad()
def vanilla_tokens(model,image):
    """Original CLIP q-k transformer, expose all patch tokens rather than CLS only."""
    v=model.visual;x=v.conv1(image.to(v.conv1.weight.dtype));b,c,h,w=x.shape
    x=x.flatten(2).transpose(1,2)
    x=torch.cat((v.class_embedding.to(x.dtype)[None,None].expand(b,1,-1),x),1)
    pos=v.positional_embedding;side=int((len(pos)-1)**.5)
    patches=F.interpolate(pos[1:].reshape(side,side,-1).permute(2,0,1)[None],(h,w),mode='bilinear',align_corners=False)
    pos=torch.cat((pos[:1],patches[0].flatten(1).T),0)
    x=v.ln_pre(x+pos.to(x.dtype)).permute(1,0,2);att=[]
    for block in v.transformer.resblocks:
        q=block.ln_1(x);out,a=block.attn(q,q,q,need_weights=True)
        x=x+out;x=x+block.mlp(block.ln_2(x));att.append(a)
    features=v.ln_post(x.permute(1,0,2))@v.proj
    features=features/features.norm(dim=1,keepdim=True).clamp_min(1e-8)
    return features,torch.stack(att)

@torch.no_grad()
def run(args):
    sys.path.insert(0,str(Path(args.code_root).resolve()))
    import clip
    from datasets.transforms import normalize_img
    from utils.affutils import refine_cams_with_aff,refine_cams_with_bkg_weclip
    from utils.PAR import PAR
    from research.model_factory import build_model
    from research.train import load_heads
    cfg=json.loads(Path(args.config).read_text());os.environ['CLIP_CACHE_DIR']=cfg['clip_cache']
    train_ids,_=validate_full_splits(cfg)
    ids=read_ids(args.ids);expected_n=len(train_ids)
    if set(ids)!=set(train_ids):
        raise ValueError('CAM and seed ablations require the complete training split')
    nc=spec(cfg)['classes'];size=cfg['crop'];grid=size//16
    model=build_model(cfg).cuda().eval()
    if args.checkpoint:
        loaded=load_heads(model,args.checkpoint)
        if loaded['config'] != cfg:
            raise ValueError('Checkpoint configuration differs from this full-data run')
    if args.limit:ids=ids[:args.limit]
    visual=json.loads(Path(args.visual_ids).read_text())
    visual=set(visual['cam_ids'])
    tag_dict=np.load(cfg['cls_labels'],allow_pickle=True).item()
    variants=[args.name]
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    stages=['direct_cam','static_affinity_par_seed','affinity_seed','affinity_par_seed']
    matrices={v:{s:ConfusionStore(out/f'{v}_{s}_matrices.npz',len(ids),nc) for s in stages} for v in variants}
    totals={v:{s:np.zeros((nc,nc),np.int64) for s in stages} for v in variants}
    par=PAR(num_iter=20,dilations=[1,2,4,8,12,24]).cuda();started=time.time()
    for index,name in enumerate(ids):
        image=np.asarray(Image.open(image_path(cfg,name,'train')).convert('RGB'))
        gt=load_eval_mask(cfg,name,'train')
        inputs=torch.from_numpy(normalize_img(image).transpose(2,0,1).copy())[None].cuda()
        inputs=F.interpolate(inputs,(size,size),mode='bilinear',align_corners=False)
        tags=torch.tensor(tag_dict[name],device='cuda',dtype=torch.float32)
        result=model(inputs);cams=result['cams']
        records=[(args.name,cams[0],result['attention'][:,0],result['affinity'].detach(),result.get('dense'))]
        for variant,cam,attention,aff,dense in records:
            cls=tags.nonzero().flatten();keys=torch.cat((cls.new_zeros(1),cls+1))
            if cls.numel()==0:
                pred=np.zeros(gt.shape,dtype=np.uint8)
                for stage in matrices[variant]:
                    matrix=confusion(gt,pred,nc)
                    matrices[variant][stage].append(matrix)
                    totals[variant][stage]+=matrix
                continue
            direct=cam.T.reshape(nc-1,grid,grid)[cls].float()
            direct=direct-direct.flatten(1).amin(1)[:,None,None]
            direct=direct/direct.flatten(1).amax(1)[:,None,None].clamp_min(1e-6)
            direct=F.interpolate(direct[None],gt.shape,mode='bilinear',align_corners=False)[0]
            direct_full=torch.cat((1-direct.amax(0,keepdim=True),direct),0)
            pred_direct=keys[direct_full.argmax(0)].cpu().numpy()
            static_maps,static_classes=refine_cams_with_aff(cam,attention,tags,(size,size),caa_thre=.79,seg_attn=None)
            static_label,_=refine_cams_with_bkg_weclip(static_maps,inputs[0],static_classes.cpu(),par,gt.shape)
            pred_static=static_label[0].cpu().numpy()
            refined,classes=refine_cams_with_aff(cam,attention,tags,(size,size),caa_thre=.79,seg_attn=aff)
            if args.family=='ours' and cfg.get('graph'):
                from research.methods import anchored_graph
                maps=inputs.new_zeros((1,nc-1,grid,grid))
                for j,k in enumerate(classes):maps[0,k]=refined[j]
                maps/=maps.flatten(2).amax(-1)[:,:,None,None].clamp_min(1e-6)
                maps,_=anchored_graph(maps,dense,tags[None])
                refined=list(maps[0,classes])
            label,prob=refine_cams_with_bkg_weclip(refined,inputs[0],classes.cpu(),par,gt.shape)
            key=torch.cat((classes.new_zeros(1),classes+1)).to(prob.device)
            pred_aff=key[prob.argmax(0)].cpu().numpy();pred_par=label[0].cpu().numpy()
            for stage,pred in [('direct_cam',pred_direct),('static_affinity_par_seed',pred_static),('affinity_seed',pred_aff),('affinity_par_seed',pred_par)]:
                matrix=confusion(gt,pred,nc)
                matrices[variant][stage].append(matrix)
                totals[variant][stage]+=matrix
            if name in visual:
                folder=out/'visual_raw'/variant;folder.mkdir(parents=True,exist_ok=True)
                Image.fromarray(image).save(folder/f'{name}_image.png');Image.fromarray(gt).save(folder/f'{name}_gt.png')
                for stage,pred in [('direct',pred_direct),('static_par',pred_static),('affinity',pred_aff),('par',pred_par)]:Image.fromarray(pred.astype(np.uint8)).save(folder/f'{name}_{stage}.png')
                coarse=cam.T.reshape(nc-1,grid,grid).float().cpu().numpy()
                np.savez_compressed(folder/f'{name}_cams.npz',cams=coarse,classes=classes.cpu().numpy(),normalized_refined=F.interpolate(prob[None],(80,80),mode='bilinear',align_corners=False)[0].cpu().numpy())
        if (index+1)%50==0:
            progress={'status':'running','images':index+1,'n':len(ids),'seconds':time.time()-started}
            dump(out/'status.json',progress);print(json.dumps(progress),flush=True)
    results={}
    for variant in variants:
        results[variant]={}
        for stage,values in matrices[variant].items():
            values.export(ids)
            results[variant][stage]=score(totals[variant][stage][None]) if not args.limit else {'smoke_only':True,'miou_percent':None}
    summary={'status':'completed' if not args.limit else 'smoke_only','n':len(ids),'expected_full_n':expected_n,'dataset':cfg['dataset'],'num_classes':nc,'train_fraction':1.0,
             'split_sha256':sha(args.ids),'source_checkpoint':args.checkpoint,'checkpoint_sha256':sha(args.checkpoint) if args.checkpoint else None,
             'family':args.family,'class_tags_used':True,'dense_crf':False,'paper_seed_column':'affinity_par_seed',
             'input_size':size,'results':results,'seconds':time.time()-started,'peak_mib':torch.cuda.max_memory_allocated()/2**20}
    dump(out/'summary.json',summary);dump(out/'status.json',{'status':summary['status'],'images':len(ids)});print(json.dumps({'completed':args.name,'n':len(ids)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--code-root',required=True);p.add_argument('--config',required=True)
    p.add_argument('--ids',required=True);p.add_argument('--visual-ids',required=True);p.add_argument('--out',required=True)
    p.add_argument('--family',choices=['ours'],default='ours');p.add_argument('--name',required=True)
    p.add_argument('--checkpoint');p.add_argument('--limit',type=int,default=0)
    args=p.parse_args();run(args)
