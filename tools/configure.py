"""Update resource locations in the release's JSON configurations."""
from pathlib import Path
import argparse,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--voc-root');p.add_argument('--coco-root');p.add_argument('--clip-cache')
    p.add_argument('--coco-attributes');p.add_argument('--coco-attr-clusters',type=int)
    p.add_argument('--dino-repo');p.add_argument('--dino-weights')
    p.add_argument('--config',help='One config; by default update every training config.')
    a=p.parse_args()
    changes={key:value for key,value in
             [('clip_cache',a.clip_cache),
              ('dino_repo',a.dino_repo),('dino_weights',a.dino_weights)] if value is not None}
    if not changes and not any([a.voc_root,a.coco_root,a.coco_attributes,a.coco_attr_clusters]):p.error('Supply at least one resource location.')
    candidates=[Path(a.config)] if a.config else sorted((ROOT/'research/configs').rglob('*.json'))
    for path in candidates:
        path=path.resolve()
        if not path.is_relative_to(ROOT):p.error('Configuration must be inside this release.')
        cfg=json.loads(path.read_text())
        if 'data' not in cfg:continue
        cfg.update(changes)
        if cfg.get('dataset','voc2012')=='voc2012' and a.voc_root:cfg['data']=a.voc_root
        if cfg.get('dataset')=='coco2014':
            if a.coco_root:cfg['data']=a.coco_root
            if a.coco_attributes:cfg['attributes_json']=a.coco_attributes
            if a.coco_attr_clusters:cfg['attr_clusters']=a.coco_attr_clusters
        path.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
        print(path.relative_to(ROOT))
if __name__=='__main__':main()
