"""Image-only training augmentation. Dense ground truth is inaccessible here."""
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from datasets import transforms
from research.protocol import image_path,mask_path,spec


class ImageLabelDataset(Dataset):
    def __init__(self, data, ids, labels, crop, dataset="voc2012", num_classes=21, config=None):
        self.data=Path(data); self.ids=Path(ids).read_text().split(); self.crop=crop
        self.labels=np.load(labels,allow_pickle=True).item()
        self.config=dict(config or {}, data=str(data), dataset=dataset)
        for name in self.ids:
            tag=np.asarray(self.labels[name])
            if tag.shape != (num_classes-1,) or not np.isin(tag, [0,1]).all():
                raise ValueError(f"Invalid image-level tag vector: {name}")

    def __len__(self): return len(self.ids)

    def __getitem__(self,i):
        name=self.ids[i]
        image=np.asarray(Image.open(image_path(self.config,name,'train')).convert('RGB'))
        image=transforms.random_scaling(image,scale_range=[.5,2.0])
        if random.random()<.5: image=np.fliplr(image).copy()
        image,box=transforms.random_crop(image,label=None,crop_size=self.crop,mean_rgb=[123.675,116.28,103.53])
        image=transforms.normalize_img(image).transpose(2,0,1).copy()
        # Explicit padding mask, independent of segmentation labels.
        return name,torch.from_numpy(image),torch.tensor(self.labels[name],dtype=torch.float32),torch.tensor(box)


def load_eval_image(data,name,config=None,split="val"):
    raw=np.asarray(Image.open(image_path(dict(config or {},data=str(data)),name,split)).convert('RGB'))
    normalized=transforms.normalize_img(raw).transpose(2,0,1).copy()
    return raw,torch.from_numpy(normalized)[None]


def load_eval_mask(config,name,split='val'):
    label=np.asarray(Image.open(mask_path(config,name,split)))
    nc=spec(config)['classes']
    if label.ndim!=2 or not np.isin(label,list(range(nc))+[255]).all():
        raise ValueError(f'Expected contiguous semantic mask IDs 0..{nc-1} or 255: {name}')
    return label
