"""Prepare ALL COCO 2014 images and image tags, without geometry supervision."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from research.protocol import COCO_CATEGORY_IDS, coco_annotation


def prepare(data, folder):
    data, folder = Path(data), Path(folder)
    parsed = {split: coco_annotation(data / 'annotations' / f'instances_{split}2014.json', split)
              for split in ('train', 'val')}
    numeric = [{x['id'] for x in parsed[split][1]} for split in ('train', 'val')]
    if numeric[0] & numeric[1]:
        raise ValueError('COCO training and validation image IDs overlap')
    mapping = {cat: i for i, cat in enumerate(COCO_CATEGORY_IDS)}
    obj, images, _ = parsed['train']
    names = {item['id']: Path(item['file_name']).stem for item in images}
    labels = {name: np.zeros(80, dtype=np.uint8) for name in names.values()}
    # Retain zero-tag images; annotation geometry is never used for training.
    for ann in obj['annotations']:
        labels[names[ann['image_id']]][mapping[ann['category_id']]] = 1
    folder.mkdir(parents=True, exist_ok=True)
    for split, (_, _, ids) in parsed.items():
        (folder / f'{split}_full.txt').write_text('\n'.join(ids) + '\n', encoding='utf-8')
    np.save(folder / 'cls_labels_onehot.npy', labels)
    report = {'dataset': 'coco2014', 'selection': 'all official images; no annotation/tag filtering',
              'train_count': len(parsed['train'][2]), 'val_count': len(parsed['val'][2]),
              'train_fraction': 1.0, 'train_val_overlap': 0,
              'zero_tag_train_count': sum(not value.any() for value in labels.values()),
              'category_ids_in_channel_order': list(COCO_CATEGORY_IDS),
              'mask_labels': '0 background; sorted category IDs mapped to 1..80; 255 ignore',
              'annotation_sha256': {split: hashlib.sha256((data / 'annotations' / f'instances_{split}2014.json').read_bytes()).hexdigest()
                                    for split in ('train', 'val')},
              'sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in ('train_full.txt', 'val_full.txt', 'cls_labels_onehot.npy')}}
    (folder / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (folder / 'visual_ids.json').write_text(json.dumps({'cam_ids': parsed['train'][2][:4],
        'val_ids': parsed['val'][2][:8]}, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--out', default='research/splits/coco')
    a = p.parse_args()
    print(json.dumps(prepare(a.data, a.out), indent=2))


if __name__ == '__main__':
    main()
