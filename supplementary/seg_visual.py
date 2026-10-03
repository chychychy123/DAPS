from pathlib import Path
import argparse,json,os,sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from research.protocol import mask_path,spec

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--code-root',required=True);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--visual-ids',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    sys.path.insert(0,a.code_root)
    from research.model_factory import build_model
    from research.train import load_heads
    from research.data import load_eval_image,load_eval_mask
    cfg=json.loads(Path(a.config).read_text());os.environ['CLIP_CACHE_DIR']=cfg['clip_cache']
    model=build_model(cfg).cuda().eval();load_heads(model,a.checkpoint)
    ids=json.loads(Path(a.visual_ids).read_text())['val_ids'];out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for name in ids:
        raw,x=load_eval_image(cfg['data'],name,cfg)
        logits=model(F.interpolate(x.cuda(),(cfg['crop'],cfg['crop']),mode='bilinear',align_corners=False))['seg']
        prob=F.interpolate(logits,raw.shape[:2],mode='bilinear',align_corners=False).softmax(1)[0]
        entropy=-(prob*prob.clamp_min(1e-8).log()).sum(0)/np.log(spec(cfg)['classes'])
        pred=prob.argmax(0).cpu().numpy().astype(np.uint8)
        gt=load_eval_mask(cfg,name)
        Image.fromarray(raw).save(out/f'{name}_image.png');Image.fromarray(gt).save(out/f'{name}_gt.png');Image.fromarray(pred).save(out/f'{name}_prediction.png')
        np.savez_compressed(out/f'{name}_uncertainty.npz',entropy=entropy.cpu().numpy().astype(np.float16),confidence=prob.amax(0).cpu().numpy().astype(np.float16))
    (out/'manifest.json').write_text(json.dumps({'ids':ids,'n':len(ids),'selection':'fixed first 8 official val IDs',
         'model_input':'image only; no validation image-class labels','checkpoint':a.checkpoint,'mode':'single scale 320; no CRF','purpose':'visualization, not subset-score selection'},indent=2))
    print(json.dumps({'visual_images':len(ids)}))
if __name__=='__main__':main()
