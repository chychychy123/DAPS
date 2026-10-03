"""CPU regression tests for full-dataset membership and VOC/COCO behavior."""
from pathlib import Path
import json
import subprocess
import sys
from unittest.mock import patch
import numpy as np
import pytest
import torch
from PIL import Image
from research import protocol
from research.prepare_coco import prepare
from research.data import ImageLabelDataset
from research.methods import anchored_graph, partial_label_loss
from utils.affutils import refine_cams_with_bkg_weclip

ROOT = Path(__file__).resolve().parents[1]


def config(dataset='voc2012'):
    sub = 'coco/' if dataset == 'coco2014' else ''
    return json.loads((ROOT / f'research/configs/{sub}B1_partial_labels.json').read_text())


@pytest.fixture
def voc_root(tmp_path):
    folder = tmp_path / 'ImageSets/Segmentation'
    folder.mkdir(parents=True)
    (folder / 'val.txt').write_bytes((ROOT / 'datasets/voc/val.txt').read_bytes())
    return tmp_path


def test_every_ablation_uses_full_splits(voc_root):
    for dataset in ('voc2012', 'coco2014'):
        sub = 'coco' if dataset == 'coco2014' else ''
        configs = [json.loads(path.read_text()) for path in (ROOT / 'research/configs' / sub).glob('*.json')]
        configs = [cfg for cfg in configs if 'data' in cfg]
        assert len(configs) == 5
        schedules = set()
        for cfg in configs:
            assert cfg['dataset'] == dataset and cfg['train_fraction'] == 1.0
            assert cfg['train_ids'].endswith('/train_full.txt')
            assert cfg['cam_ids'] == cfg['train_ids']
            assert cfg['num_classes'] == protocol.SPECS[dataset]['classes']
            schedules.add(tuple(cfg[k] for k in ('iters','batch','warmup','partial_start','learned_affinity_start')))
            assert cfg['iters'] >= (protocol.SPECS[dataset]['train'] + cfg['batch']-1)//cfg['batch']
            if dataset == 'voc2012':
                cfg['data'] = str(voc_root)
                train, val = protocol.validate_full_splits(cfg)
                assert len(train) == 10582 and len(val) == 1449
                assert not set(train) & set(val)
        assert len(schedules) == 1


def test_reject_reduced_duplicate_and_wrong_membership(voc_root, tmp_path):
    cfg = dict(config(), data=str(voc_root))
    full = protocol.read_ids(cfg['train_ids'])
    altered = tmp_path / 'altered.txt'
    cfg['train_ids'] = str(altered)
    for bad in (full[:len(full)//2], full[:-1]+[full[0]], full[:-1]+['2099_999999']):
        altered.write_text('\n'.join(bad))
        with pytest.raises(ValueError):
            protocol.validate_full_splits(cfg)


def make_coco_annotations(root, ntrain, nval):
    folder = root / 'annotations'
    folder.mkdir()
    for split, count, offset in [('train',ntrain,1),('val',nval,ntrain+1)]:
        images = [{'id':i,'file_name':f'COCO_{split}2014_{i:012d}.jpg'} for i in range(offset,offset+count)]
        # One positive image; all remaining zero-tag images must be retained.
        obj = {'images':images, 'categories':[{'id':i} for i in protocol.COCO_CATEGORY_IDS],
               'annotations':[{'image_id':offset,'category_id':90}]}
        (folder/f'instances_{split}2014.json').write_text(json.dumps(obj))


def test_coco_actual_full_counts_and_zero_tag_retention(tmp_path):
    make_coco_annotations(tmp_path,82783,40504)
    folder = tmp_path/'splits'
    report = prepare(tmp_path,folder)
    assert report['train_count'] == 82783 and report['val_count'] == 40504
    assert report['zero_tag_train_count'] == 82782
    labels=np.load(folder/'cls_labels_onehot.npy',allow_pickle=True).item()
    first='COCO_train2014_000000000001'
    assert labels[first].shape == (80,) and labels[first][79] == 1
    assert labels['COCO_train2014_000000082783'].sum() == 0
    cfg=dict(config('coco2014'),data=str(tmp_path),train_ids=str(folder/'train_full.txt'),val_ids=str(folder/'val_full.txt'))
    train,val=protocol.validate_full_splits(cfg)
    assert (len(train),len(val)) == (82783,40504)
    # Equal count and a valid-looking filename cannot bypass canonical membership.
    train[-1]='COCO_train2014_999999999999'
    (folder/'train_full.txt').write_text('\n'.join(train))
    with pytest.raises(ValueError): protocol.validate_full_splits(cfg)


@pytest.mark.parametrize('dataset,nc', [('voc2012',21),('coco2014',81)])
def test_training_reads_only_rgb_and_keeps_tail_batch(tmp_path,dataset,nc):
    cfg=dict(config(dataset),data=str(tmp_path),crop=32)
    names=[f'img{i}' for i in range(5)]
    ids=tmp_path/'ids.txt';ids.write_text('\n'.join(names))
    labels=tmp_path/'labels.npy';np.save(labels,{name:np.zeros(nc-1,np.uint8) for name in names})
    for name in names:
        path=protocol.image_path(cfg,name,'train');path.parent.mkdir(parents=True,exist_ok=True)
        Image.fromarray(np.zeros((40,40,3),np.uint8)).save(path)
    ds=ImageLabelDataset(str(tmp_path),str(ids),str(labels),32,dataset,nc,cfg)
    original=Image.open
    def guarded(path,*args,**kwargs):
        assert 'SegmentationClass' not in str(path)
        return original(path,*args,**kwargs)
    with patch('PIL.Image.open',guarded):
        batches=list(torch.utils.data.DataLoader(ds,batch_size=4,drop_last=False))
    assert [len(x[0]) for x in batches]==[4,1]
    assert batches[0][2].shape==(4,nc-1)
    assert set(name for b in batches for name in b[0])==set(names)


@pytest.mark.parametrize('nc', [21,81])
def test_losses_and_empty_foreground(nc):
    tags=torch.zeros(2,nc-1);tags[0,0]=1
    maps,proto=anchored_graph(torch.rand(2,nc-1,4,4),torch.randn(2,16,4,4),tags)
    assert maps.shape==(2,nc-1,4,4) and proto.shape==(2,nc,4,4)
    assert torch.allclose(proto.sum(1),torch.ones(2,4,4),atol=1e-6)
    assert maps[1].count_nonzero()==0
    logits=torch.randn(2,nc,4,4,requires_grad=True)
    target=torch.zeros(2,4,4,dtype=torch.long)
    loss,_=partial_label_loss(logits,target,proto,tags)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    target,prob=refine_cams_with_bkg_weclip([],torch.zeros(3,32,32),torch.empty(0,dtype=torch.long),None,(32,32))
    assert target.shape==(1,32,32) and target.count_nonzero()==0
    assert torch.equal(prob,torch.ones(1,32,32))


def test_coco_paths_and_attribute_isolation():
    cfg=config('coco2014');name='COCO_val2014_000000000042'
    assert str(protocol.image_path(cfg,name)).replace('\\','/').endswith(f'JPEGImages/val2014/{name}.jpg')
    assert protocol.mask_path(cfg,name).stem==name
    options=protocol.model_options(cfg)
    assert options['num_classes']==81 and options['dataset_name']=='coco'
    assert 'pascal_voc' not in options['json_file']
    with patch.object(Path,'is_file',return_value=False):
        with pytest.raises(FileNotFoundError): protocol.validate_attribute_resource(cfg)


def test_coco_launcher_resolves_full_cam_and_visual_lists(tmp_path):
    run=tmp_path/'run';run.mkdir()
    cfg=config('coco2014');(run/'config.json').write_text(json.dumps(cfg))
    for task in ('cam','visualize'):
        out=subprocess.check_output([sys.executable,'tools/launch.py',task,'--run',str(run),'--out',str(tmp_path/task),'--dry-run'],cwd=ROOT,text=True)
        command=json.loads(out)['command']
        assert command[command.index('--visual-ids')+1]=='research/splits/coco/visual_ids.json'
        if task=='cam': assert command[command.index('--ids')+1]=='research/splits/coco/train_full.txt'


@pytest.mark.parametrize('nc', [21,81])
def test_evaluation_confusions_use_all_classes(nc):
    from research.final_evaluate import confusion
    from supplementary.cam_eval import confusion as cam_confusion, score
    from research.train import metric
    gt=np.arange(nc,dtype=np.uint8).reshape(1,nc)
    pred=gt.copy()
    for calculate in (confusion,cam_confusion):
        matrix=calculate(gt,pred,nc)
        assert np.array_equal(matrix,np.eye(nc,dtype=np.int64))
        assert metric(matrix)['miou_percent']==100.0
        assert score(matrix[None])['miou_percent']==100.0


def test_configure_keeps_dataset_roots_separate(tmp_path):
    import importlib.util
    module_spec=importlib.util.spec_from_file_location('configure_under_test',ROOT/'tools/configure.py')
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    for dataset,sub in [('voc2012',''),('coco2014','coco/')]:
        path=tmp_path/f'research/configs/{sub}test.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(config(dataset)))
    args=['configure.py','--voc-root','voc_full','--coco-root','coco_full']
    with patch.object(module,'ROOT',tmp_path),patch.object(sys,'argv',args):module.main()
    assert json.loads((tmp_path/'research/configs/test.json').read_text())['data']=='voc_full'
    assert json.loads((tmp_path/'research/configs/coco/test.json').read_text())['data']=='coco_full'


def test_bundled_voc_tags_cover_every_training_id():
    cfg=config()
    ds=ImageLabelDataset(cfg['data'],cfg['train_ids'],cfg['cls_labels'],cfg['crop'])
    assert len(ds)==10582


def test_confusion_storage_and_incomplete_coverage(tmp_path):
    from research.confusions import ConfusionStore
    store=ConfusionStore(tmp_path/'matrices.npz',2,81)
    matrix=np.eye(81,dtype=np.int64)
    store.append(matrix)
    with pytest.raises(ValueError):store.export(['one'])
    store.append(matrix*3);store.export(['one','two'])
    assert not (tmp_path/'matrices.working.npy').exists()
    with np.load(tmp_path/'matrices.npz') as z:
        assert z['ids'].tolist()==['one','two']
        assert z['matrices'].shape==(2,81,81)
        assert np.array_equal(z['matrices'].sum(0),matrix*4)


def test_coco_mask_ids_are_contiguous(tmp_path):
    from research.data import load_eval_mask
    cfg=dict(config('coco2014'),data=str(tmp_path));name='COCO_val2014_000000000042'
    path=protocol.mask_path(cfg,name);path.parent.mkdir(parents=True)
    Image.fromarray(np.array([[0,1,80,255]],np.uint8)).save(path)
    assert load_eval_mask(cfg,name).tolist()==[[0,1,80,255]]
    Image.fromarray(np.array([[90]],np.uint8)).save(path)
    with pytest.raises(ValueError):load_eval_mask(cfg,name)
