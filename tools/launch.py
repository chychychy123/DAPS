"""Portable CLI; imports no GPU libraries until the selected program is started."""
from pathlib import Path
import argparse,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='task',required=True)
    train=sub.add_parser('train')
    train.add_argument('--config',default='research/configs/B1_partial_labels.json')
    train.add_argument('--run-dir',required=True)
    ev=sub.add_parser('eval')
    ev.add_argument('--run',required=True);ev.add_argument('--out',required=True)
    ev.add_argument('--checkpoint',default='best.pt')
    ev.add_argument('--recipe',choices=['I0','I1'],default='I1')
    cam=sub.add_parser('cam')
    cam.add_argument('--run',required=True);cam.add_argument('--out',required=True)
    cam.add_argument('--checkpoint',default='best.pt')
    cam.add_argument('--ids',help='Defaults to the complete training list in the run configuration.')
    vis=sub.add_parser('visualize')
    vis.add_argument('--run',required=True);vis.add_argument('--out',required=True)
    vis.add_argument('--checkpoint',default='best.pt')
    for child in [train,ev,cam,vis]:
        child.add_argument('--dry-run',action='store_true',help='Print the command without executing it.')
    args=p.parse_args()
    os.chdir(ROOT)
    prefix=[sys.executable,'-m']
    if args.task=='train':
        out=Path(args.run_dir)
        command=prefix+['research.train','--config',args.config,'--run-dir',str(out)]
    elif args.task=='eval':
        out=Path(args.out)
        recipe='excel_official' if args.recipe=='I1' else 'initial'
        scales=['0.7','1.0','1.2','1.5'] if args.recipe=='I1' else ['0.75','1.0']
        command=prefix+['research.final_evaluate','--run',args.run,
          '--out',str(out),'--checkpoint',args.checkpoint,'--scales',*scales,
          '--flip','--crf','--recipe',recipe]
    else:
        out=Path(args.out);run=Path(args.run)
        cfg=json.loads((run/'config.json').read_text()) if (run/'config.json').is_file() else {}
        common=['--code-root',str(ROOT),'--config',str(run/'config.json'),
                '--checkpoint',str(run/args.checkpoint),
                '--visual-ids',cfg.get('visual_ids','research/splits/visual_ids.json'),'--out',str(out)]
        if args.task=='cam':
            name='model'
            if (run/'config.json').is_file():
                name=json.loads((run/'config.json').read_text()).get('experiment','model')
            ids=args.ids or cfg.get('cam_ids',cfg.get('train_ids','research/splits/train_full.txt'))
            command=prefix+['supplementary.cam_eval',*common,'--ids',ids,
                            '--family','ours','--name',name]
        else:command=prefix+['supplementary.seg_visual',*common]
    if args.dry_run:
        print(json.dumps({'cwd':str(ROOT),'command':command},indent=2));return
    if out.exists():raise SystemExit(f'Choose a new output directory; already exists: {out}')
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        env.setdefault(key,'4')
    subprocess.run(command,cwd=ROOT,env=env,check=True)
if __name__=='__main__':main()
