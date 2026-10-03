"""Shared full-dataset contracts, without importing GPU libraries."""
from pathlib import Path
import json

SPECS = {'voc2012': {'train': 10582, 'val': 1449, 'classes': 21},
         'coco2014': {'train': 82783, 'val': 40504, 'classes': 81}}
COCO_CATEGORY_IDS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
    76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90)


def spec(cfg):
    return SPECS[cfg.get('dataset', 'voc2012')]


def read_ids(path):
    ids = Path(path).read_text(encoding='utf-8').split()
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate image IDs: {path}')
    if any(Path(name).name != name or '/' in name or '\\' in name for name in ids):
        raise ValueError(f'Expected image stems, not paths: {path}')
    return ids


def coco_annotation(path, split):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if tuple(sorted(c['id'] for c in obj['categories'])) != COCO_CATEGORY_IDS:
        raise ValueError('Expected the 80 official COCO instance categories')
    images = sorted(obj['images'], key=lambda item: item['id'])
    ids = [Path(item['file_name']).stem for item in images]
    expected = SPECS['coco2014'][split]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f'Expected all {expected} COCO {split} images, got {len(ids)}')
    if len({item['id'] for item in images}) != expected:
        raise ValueError('Duplicate numeric COCO image IDs')
    for item, name in zip(images, ids):
        if name != f"COCO_{split}2014_{item['id']:012d}" or item['file_name'] != name + '.jpg':
            raise ValueError(f'Unexpected official COCO 2014 filename: {item["file_name"]}')
    return obj, images, ids


def canonical_ids(cfg, split):
    if cfg.get('dataset', 'voc2012') == 'coco2014':
        return coco_annotation(Path(cfg['data']) / 'annotations' / f'instances_{split}2014.json', split)[2]
    if split == 'train':
        return read_ids(Path(__file__).resolve().parents[1] / 'datasets/voc/train_aug.txt')
    ids = read_ids(Path(cfg['data']) / 'ImageSets/Segmentation/val.txt')
    bundled = read_ids(Path(__file__).resolve().parents[1] / 'datasets/voc/val.txt')
    if set(ids) != set(bundled):
        raise ValueError('VOC validation list differs from the bundled official split')
    return ids


def validate_full_splits(cfg):
    expected = spec(cfg)
    if cfg.get('num_classes', 21) != expected['classes']:
        raise ValueError('Dataset and segmentation class count disagree')
    lists = {}
    for split in ('train', 'val'):
        ids = read_ids(cfg[f'{split}_ids'])
        canonical = canonical_ids(cfg, split)
        if len(ids) != expected[split] or len(canonical) != expected[split] or set(ids) != set(canonical):
            raise ValueError(f'{split} must contain all {expected[split]} official protocol IDs')
        lists[split] = ids
    overlap = set(lists['train']) & set(lists['val'])
    if cfg.get('dataset') == 'coco2014':
        overlap |= {x.rsplit('_', 1)[1] for x in lists['train']} & {x.rsplit('_', 1)[1] for x in lists['val']}
    if overlap:
        raise ValueError('Training and validation IDs overlap')
    return lists['train'], lists['val']


def image_path(cfg, name, split='val'):
    folder = 'JPEGImages' if cfg.get('dataset', 'voc2012') == 'voc2012' else f'JPEGImages/{split}2014'
    return Path(cfg['data']) / cfg.get(f'{split}_image_dir', folder) / f'{name}.jpg'


def mask_path(cfg, name, split='val'):
    folder = 'SegmentationClassAug' if split == 'train' else 'SegmentationClass'
    if cfg.get('dataset') == 'coco2014':
        folder = 'SegmentationClass/coco_seg_anno'
    return Path(cfg['data']) / cfg.get(f'{split}_mask_dir', folder) / f'{name}.png'


def model_options(cfg):
    classes = spec(cfg)['classes']
    dataset = 'pascal_voc' if classes == 21 else 'coco'
    return {'num_classes': classes, 'dataset_name': dataset,
            'num_atrr_clusters': cfg.get('attr_clusters', 112),
            'json_file': cfg.get('attributes_json',
                f'attributes_text/descriptors_{dataset}_gpt4.0_cluster_a_photo_of4.json')}


def validate_attribute_resource(cfg):
    options = model_options(cfg)
    bank = Path('attributes_text') / (f"{options['dataset_name']}_desc_clip_ViT-B-16_gpt4.0_cluster_"
                                     f"{options['num_atrr_clusters']}_embedding_bank.pth")
    if not bank.is_file() and not Path(options['json_file']).is_file():
        raise FileNotFoundError(f"Supply matching {cfg.get('dataset', 'voc2012')} attributes: "
                                f"{bank} or {options['json_file']}. No VOC fallback is allowed.")
