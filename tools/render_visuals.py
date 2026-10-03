"""Render the archived single-scale visualization format into aligned panels."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def palette():
    lut=np.zeros((256,3),np.uint8)
    for i in range(256):
        value=i
        for bit in range(8):
            for c in range(3):lut[i,c]|=((value>>c)&1)<<(7-bit)
            value>>=3
    lut[255]=255
    return lut
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.input);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    images=sorted(root.glob('*_image.png'))
    if not images:raise SystemExit('No *_image.png files found.')
    lut=palette()
    for page,start in enumerate(range(0,len(images),4),1):
        group=images[start:start+4]
        fig,axes=plt.subplots(len(group),5,figsize=(12,2.25*len(group)),squeeze=False)
        for row,file in enumerate(group):
            name=file.name.removesuffix('_image.png')
            rgb=np.asarray(Image.open(file).convert('RGB'))
            gt=np.asarray(Image.open(root/f'{name}_gt.png'))
            pred=np.asarray(Image.open(root/f'{name}_prediction.png'))
            with np.load(root/f'{name}_uncertainty.npz') as z:entropy=z['entropy']
            valid=gt!=255;error=(rgb*.35).astype(np.uint8)
            error[valid&(gt==0)&(pred!=0)]=[230,55,45]
            error[valid&(gt!=0)&(pred==0)]=[40,105,235]
            error[valid&(gt!=0)&(pred!=0)&(gt!=pred)]=[245,200,30]
            error[~valid]=255
            for col,array in enumerate([rgb,lut[gt],lut[pred],error,entropy]):
                ax=axes[row,col]
                ax.imshow(array,cmap='magma' if col==4 else None,
                          **({'vmin':0,'vmax':1} if col==4 else {}))
                ax.set_box_aspect(.75);ax.set_facecolor('#f1f1f1')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_linewidth(.35);spine.set_color('#ccc')
                if row==0:ax.set_title(['Image','Ground truth','Prediction','Errors','Entropy'][col],fontsize=9)
            axes[row,0].set_ylabel(name,fontsize=7)
        fig.tight_layout(pad=.8)
        fig.savefig(out/f'visualization_{page}.png',dpi=300)
        fig.savefig(out/f'visualization_{page}.pdf')
        plt.close(fig)
    print(out)
if __name__=='__main__':main()
