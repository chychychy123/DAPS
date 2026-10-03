"""Separate full-val single/multiscale/CRF scores with checkpoint provenance.

This module is never imported by the training queue. It must run after a job
completes, against a fixed checkpoint and the original run configuration.
"""
from pathlib import Path
import argparse,json,os,time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from research.model_factory import build_model
from research.protocol import validate_full_splits,spec,mask_path
from research.data import load_eval_image,load_eval_mask
from research.confusions import ConfusionStore
from research.train import load_heads,metric,sha,write_json
from research.release_version import source_revision


def confusion(gt,pred,num_classes=21):
    valid=(gt>=0)&(gt<num_classes)
    return np.bincount(num_classes*gt[valid].astype(np.int64)+pred[valid],minlength=num_classes**2).reshape(num_classes,num_classes)


def crf(image,probs,recipe='initial'):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    nc,h,w=probs.shape
    d=dcrf.DenseCRF2D(w,h,nc)
    d.setUnaryEnergy(np.array(unary_from_softmax(probs),dtype=np.float32,order='C',copy=True))
    # Explicit writable buffers also support old Cython builds.
    pos_xy,bi_xy,bi_rgb,bi_weight=(1,67,3,4) if recipe=='excel_official' else (3,80,13,10)
    d.addPairwiseGaussian(sxy=pos_xy,compat=3)
    d.addPairwiseBilateral(sxy=bi_xy,srgb=bi_rgb,rgbim=np.array(image,dtype=np.uint8,order='C',copy=True),compat=bi_weight)
    return np.asarray(d.inference(10)).reshape(nc,h,w).argmax(0)


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--checkpoint',default='best.pt')
    p.add_argument('--out',required=True);p.add_argument('--scales',nargs='+',type=float,default=[.75,1.])
    p.add_argument('--flip',action='store_true');p.add_argument('--crf',action='store_true')
    p.add_argument('--recipe',choices=['initial','excel_official'],default='initial')
    p.add_argument('--smoke',action='store_true');a=p.parse_args()
    run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((run/'config.json').read_text());provenance=json.loads((run/'provenance.json').read_text())
    assert sha(cfg['dino_weights'])==provenance['dino_sha256']
    assert sha(Path(cfg['clip_cache'])/'ViT-B-16.pt')==provenance['clip_sha256']
    assert sha(cfg['train_ids'])==provenance['train_sha256'] and sha(cfg['val_ids'])==provenance['val_sha256']
    _,ids=validate_full_splits(cfg)
    nc=spec(cfg)['classes']
    os.environ['CLIP_CACHE_DIR']=cfg['clip_cache'];model=build_model(cfg).cuda().eval()
    ckpt=run/a.checkpoint;loaded=load_heads(model,ckpt)
    if loaded['config'] != cfg:
        raise ValueError('Checkpoint training configuration differs from this full-data run')
    if a.recipe=='excel_official':
        # ExCEL's official evaluator retains the original frozen CLIP positions
        # and interpolates them independently at each inference resolution.
        original=torch.jit.load(str(Path(cfg['clip_cache'])/'ViT-B-16.pt'),map_location='cpu')
        position=original.state_dict()['visual.positional_embedding'].to(model.encoder.visual.positional_embedding)
        model.encoder.visual.positional_embedding=torch.nn.Parameter(position.clone(),requires_grad=False)
        del original
    names=['single','multiscale']+(['multiscale_crf'] if a.crf else [])
    evaluated_ids=ids[:2] if a.smoke else ids
    hist={key:np.zeros((nc,nc),np.int64) for key in names}
    each={key:ConfusionStore(out/f'{key}_confusions.npz',len(evaluated_ids),nc) for key in names}
    per_image=[];started=time.time()
    for index,name in enumerate(evaluated_ids):
        raw,x=load_eval_image(cfg['data'],name,cfg);x=x.cuda()
        label=load_eval_mask(cfg,name)
        base=F.interpolate(x,(cfg['crop'],cfg['crop']),mode='bilinear',align_corners=False)
        base_logit=model(base)['seg'].float()
        base_probability=F.interpolate(base_logit,label.shape,mode='bilinear',align_corners=False).softmax(1)
        probabilities=[];resized_logits=[]
        scales=([1.]+[s for s in a.scales if s!=1.]) if a.recipe=='excel_official' else a.scales
        for scale in scales:
            size=max(32,int(round(cfg['crop']*scale/16))*16)
            image=F.interpolate(x,(size,size),mode='bilinear',align_corners=False)
            logits=base_logit if size==cfg['crop'] else model(image)['seg'].float()
            if a.flip and not (a.recipe=='excel_official' and scale==1.):
                logits=(logits+model(image.flip(-1))['seg'].float().flip(-1))*.5
            resized=F.interpolate(logits,label.shape,mode='bilinear',align_corners=False)
            resized_logits.append(resized);probabilities.append(resized.softmax(1))
        aggregated=torch.stack(resized_logits).mean(0).softmax(1) if a.recipe=='excel_official' else torch.stack(probabilities).mean(0)
        probability=aggregated[0].cpu().numpy()
        predictions={'single':base_probability.argmax(1)[0].cpu().numpy(),
                     'multiscale':probability.argmax(0)}
        if a.crf:predictions['multiscale_crf']=crf(raw,probability,a.recipe)
        for key,pred in predictions.items():
            mat=confusion(label,pred,nc);hist[key]+=mat;each[key].append(mat)
        per_image.append(name)
        if (index+1)%100==0:print(json.dumps({'images':index+1,'wall_seconds':time.time()-started}),flush=True)
        # Predetermined examples, not selected by prediction quality.
        if index<8:
            (out/'examples').mkdir(exist_ok=True)
            Image.fromarray(raw).save(out/'examples'/f'{name}_image.png')
            Image.fromarray(label).save(out/'examples'/f'{name}_gt.png')
            for key,pred in predictions.items():Image.fromarray(pred.astype(np.uint8)).save(out/'examples'/f'{name}_{key}.png')
    metadata={'dataset':cfg['dataset'],'num_classes':nc,'train_n':provenance['train_n'],'train_fraction':1.0,
              'step':loaded['step'],'n':len(per_image),'scales':a.scales,'flip':a.flip,
              'image_label_masking':False,'val_sha256':provenance['val_sha256'],
              'checkpoint_sha256':sha(ckpt),'training_code_commit':provenance['code_commit'],
              'dino_sha256':provenance['dino_sha256'],'wall_seconds':time.time()-started,
              'peak_mib':torch.cuda.max_memory_allocated()/2**20,'smoke_only':a.smoke}
    metadata.update(recipe=a.recipe,aggregation='mean_logits_then_softmax' if a.recipe=='excel_official' else 'mean_probabilities',
                    base_scale_flip=a.flip and a.recipe!='excel_official',
                    crf_parameters={'iterations':10,'pos_xy':1 if a.recipe=='excel_official' else 3,'pos_w':3,
                                    'bi_xy':67 if a.recipe=='excel_official' else 80,'bi_rgb':3 if a.recipe=='excel_official' else 13,
                                    'bi_w':4 if a.recipe=='excel_official' else 10},
                    inference_source_commit=source_revision())
    all_results={}
    for key in names:
        result={'miou_percent':None} if a.smoke else metric(hist[key])
        result.update(metadata,crf=key.endswith('_crf'),evaluation=key)
        if key=='single':result.update(scales=[1.],flip=False)
        write_json(out/f'{key}.json',result)
        each[key].export(per_image)
        all_results[key]=result['miou_percent']
    write_json(out/'summary.json',dict(metadata,results=all_results))
    print(json.dumps(all_results),flush=True)

if __name__=='__main__':main()
