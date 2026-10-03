"""Independently verify full-val confusion sums and official mask denominators."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image
from research.protocol import canonical_ids, spec, mask_path

def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation',required=True);p.add_argument('--data',required=True)
    p.add_argument('--config',help='Optional run config for custom mask paths');a=p.parse_args()
    report=Path(a.evaluation); obj=json.loads(report.read_text())
    cfg=json.loads(Path(a.config).read_text()) if a.config else {'dataset':obj.get('dataset','voc2012')}
    cfg['data']=a.data
    nc=spec(cfg)['classes'];n=spec(cfg)['val']
    assert obj['n']==n and not obj.get('smoke_only',False)
    assert obj['image_label_masking'] is False
    matrices=report.with_name(report.stem+'_confusions.npz')
    with np.load(matrices,allow_pickle=False) as z: ids=z['ids'].tolist(); m=z['matrices']
    expected=canonical_ids(cfg,'val')
    assert len(ids)==len(set(ids))==n and set(ids)==set(expected)
    assert m.shape==(n,nc,nc) and (m>=0).all()
    for name,conf in zip(ids,m):
        gt=np.asarray(Image.open(mask_path(cfg,name)))
        assert np.isin(gt,list(range(nc))+[255]).all(),name
        count=np.bincount(gt[(gt>=0)&(gt<nc)].astype(np.int64),minlength=nc)
        assert np.array_equal(conf.sum(1),count),name
    hist=m.sum(0); denominator=hist.sum(0)+hist.sum(1)-hist.diagonal()
    iou=hist.diagonal()/denominator; miou=float(iou.mean()*100)
    assert np.isclose(miou,obj['miou_percent'],atol=1e-10)
    assert np.array_equal(hist,np.array(obj['confusion']))
    verification={'status':'verified','evaluation':str(report),'dataset':cfg['dataset'],'n':n,'miou_percent':miou,
                  'confusions_sha256':hashlib.sha256(matrices.read_bytes()).hexdigest(),
                  'checks':['full official unique IDs',f'per-image GT pixel count for all {nc} classes',
                            'sum of all image confusion matrices',f'independently recomputed {nc}-class mean IoU'],
                  'scope':'Arithmetic and coverage verification; cannot independently prove prediction correctness without rerunning checkpoint'}
    report.with_name(report.stem+'_verified.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification))

if __name__=='__main__':main()
