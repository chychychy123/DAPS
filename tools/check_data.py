"""Check full lists, all image/mask files, labels and attribute resources on CPU."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.protocol import validate_full_splits, spec, image_path, mask_path, validate_attribute_resource


def check(cfg, cam=False):
    train, val = validate_full_splits(cfg)
    missing = []
    total_missing = 0
    for split, ids in [('train', train), ('val', val)]:
        for name in ids:
            paths = [image_path(cfg, name, split)]
            if split == 'val' or cam:
                paths.append(mask_path(cfg, name, split))
            for path in paths:
                if not path.is_file():
                    total_missing += 1
                    if len(missing) < 10:
                        missing.append(str(path))
    if total_missing:
        raise FileNotFoundError(f'{total_missing} missing image/mask files; first examples: {missing}')
    labels = np.load(cfg['cls_labels'], allow_pickle=True).item()
    for name in train:
        tag = np.asarray(labels[name])
        if tag.shape != (spec(cfg)['classes']-1,) or not np.isin(tag, [0, 1]).all():
            raise ValueError(f'Invalid image-level labels: {name}')
    validate_attribute_resource(cfg)
    return {'status': 'passed', 'dataset': cfg['dataset'], 'train_n': len(train),
            'val_n': len(val), 'cam_masks_checked': cam, 'train_fraction': 1.0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--cam', action='store_true', help='Also require masks for ALL training images.')
    a = p.parse_args()
    os.chdir(ROOT)
    print(json.dumps(check(json.loads(Path(a.config).read_text()), a.cam), indent=2))


if __name__ == '__main__':
    main()
