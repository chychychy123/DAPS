"""Prepare complete VOC train_aug and official validation lists, without sampling."""
from pathlib import Path
import argparse, hashlib, json
from research.protocol import read_ids

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', help='VOC2012 root; omit to use bundled official val list.')
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = root / 'research/splits'
    train = read_ids(root / 'datasets/voc/train_aug.txt')
    val_path = Path(a.data) / 'ImageSets/Segmentation/val.txt' if a.data else root / 'datasets/voc/val.txt'
    val = read_ids(val_path)
    if len(train) != 10582 or len(val) != 1449 or set(train) & set(val):
        raise ValueError('Expected disjoint full VOC lists: 10582 train, 1449 val')
    folder.mkdir(parents=True, exist_ok=True)
    for name, values in [('train_full', train), ('val_full', val)]:
        (folder / f'{name}.txt').write_text('\n'.join(values) + '\n', encoding='utf-8')
    report = {'dataset': 'voc2012', 'selection': 'all train_aug images; no sampling',
              'train_count': len(train), 'val_count': len(val), 'train_fraction': 1.0,
              'train_val_overlap': 0, 'cam_ids': 'train_full.txt',
              'sha256': {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                         for name in ('train_full.txt', 'val_full.txt')}}
    (folder / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
