"""Numerical contracts that protect meaningful research claims."""
import json
from pathlib import Path
from unittest.mock import patch
import torch
from PIL import Image
from research.methods import anchored_graph,partial_label_loss
from research.data import ImageLabelDataset

def main():
    torch.manual_seed(4)
    cams=torch.rand(2,20,4,4); dense=torch.randn(2,32,4,4)
    tags=torch.zeros(2,20); tags[:,:2]=1
    out,proto=anchored_graph(cams,dense,tags)
    assert torch.isfinite(out).all() and torch.isfinite(proto).all()
    assert out[:,2:].abs().max()==0
    assert proto[:,3:].abs().max()==0
    assert torch.allclose(proto.sum(1),torch.ones(2,4,4),atol=1e-6)
    logits=torch.randn(2,21,4,4,requires_grad=True); target=torch.ones(2,4,4,dtype=torch.long)
    ignored=torch.full_like(target,255)
    loss,_=partial_label_loss(logits,ignored,proto,tags)
    assert loss.item()==0; loss.backward(); assert torch.isfinite(logits.grad).all()
    logits.grad=None
    loss,_=partial_label_loss(logits,target,proto,tags); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    # Training data must not even open a segmentation-mask file.
    cfg=json.loads(Path('research/configs/B0_dinov2_control.json').read_text())
    ds=ImageLabelDataset(cfg['data'],cfg['train_ids'],'datasets/voc/cls_labels_onehot.npy',cfg['crop'])
    original=Image.open
    def guarded(path,*args,**kwargs):
        assert 'SegmentationClass' not in str(path), 'Pixel supervision leakage'
        return original(path,*args,**kwargs)
    with patch('PIL.Image.open',guarded):
        for i in range(16):
            _,x,y,box=ds[i]; assert x.shape==(3,320,320) and y.shape==(20,)
    report={'status':'passed','checks':['finite graph','absent-class veto','probability normalization','all-ignore zero loss and finite gradients','partial loss gradients','16 real training samples without mask access']}
    Path('research/evidence').mkdir(exist_ok=True)
    Path('research/evidence/contracts.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report))

if __name__=='__main__': main()
