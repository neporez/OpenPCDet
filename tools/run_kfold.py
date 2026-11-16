"""
Run a k-fold experiment grid: every value of one setting x every fold, each run a normal train_with_val.py call.

    python run_kfold.py --group dropout --dropouts 0 0.25 0.5            # 3 x 5 folds, 3 runs at a time
    python kfold_summary.py --group dropout                             # mean / std over the folds

Runs go to output/kriso_models/kriso_voxelnext/<group>/drop<rate>_fold<k>/ (stdout in run.out). A run whose
`done` marker exists is skipped, so the grid can be restarted after an interruption (an unfinished run starts over).
Order: fold 0 of every setting first, then fold 1, ... so partial results are already comparable across settings.
Every run uses the same seed (model init, data order, augmentation): settings differ only in the setting.
Each run evaluates val and the training subset fold<k>_train_eval every --eval_interval epochs and keeps only
checkpoint_best.pth / checkpoint_last.pth (train_with_val.py --save_best_last --train_eval_split).
"""
import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
CFG = 'cfgs/kriso_models/kriso_voxelnext.yaml'
OUT = TOOLS.parent / 'output' / 'kriso_models' / 'kriso_voxelnext'


def fmt(rate):
    return ('%g' % rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, help='output sub-directory of this experiment grid')
    parser.add_argument('--dropouts', type=float, nargs='+', default=[0.0], help='MODEL.DENSE_HEAD.DROPOUT values')
    parser.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--parallel', type=int, default=3, help='runs at the same time on the GPU')
    parser.add_argument('--workers', type=int, default=4, help='dataloader workers per run')
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--extra_set', nargs='*', default=[], help='more --set KEY VALUE pairs for every run')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    jobs = []
    for fold, rate in itertools.product(args.folds, args.dropouts):
        name = 'drop%s_fold%d' % (fmt(rate), fold)
        run_dir = OUT / args.group / name
        if (run_dir / 'done').exists():
            print('skip (done)', name)
            continue
        cmd = [sys.executable, 'train_with_val.py', '--cfg_file', CFG, '--epochs', str(args.epochs),
               '--extra_tag', '%s/%s' % (args.group, name), '--workers', str(args.workers), '--seed', str(args.seed),
               '--eval_interval', str(args.eval_interval), '--save_best_last', '--train_eval_split', 'fold%d_train_eval' % fold,
               '--set', 'DATA_CONFIG.DATA_SPLIT.train', 'fold%d_train' % fold, 'DATA_CONFIG.DATA_SPLIT.test', 'fold%d_val' % fold,
               'MODEL.DENSE_HEAD.DROPOUT', repr(float(rate))] + args.extra_set  # '0.0', not '0': the cfg parser checks the type
        jobs.append((name, run_dir, cmd))
    print('%d runs to do, %d at a time' % (len(jobs), args.parallel))
    if args.dry_run:
        for name, _, cmd in jobs:
            print(name, ' '.join(cmd))
        return

    running, failed = [], []
    while jobs or running:
        while jobs and len(running) < args.parallel:
            name, run_dir, cmd = jobs.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            out = open(run_dir / 'run.out', 'w')
            running.append((name, run_dir, subprocess.Popen(cmd, cwd=TOOLS, stdout=out, stderr=subprocess.STDOUT), out, time.time()))
            print('%s start %s' % (time.strftime('%m-%d %H:%M'), name), flush=True)
            time.sleep(30)  # stagger the starts (data loading / cuda init)
        for item in list(running):
            name, run_dir, proc, out, t0 = item
            if proc.poll() is None:
                continue
            out.close()
            running.remove(item)
            if proc.returncode == 0:
                (run_dir / 'done').write_text(time.strftime('%Y-%m-%d %H:%M:%S\n'))
                print('%s done  %s (%.1f h)' % (time.strftime('%m-%d %H:%M'), name, (time.time() - t0) / 3600), flush=True)
            else:
                failed.append(name)
                print('%s FAILED %s (exit %d, see %s)' % (time.strftime('%m-%d %H:%M'), name, proc.returncode, run_dir / 'run.out'), flush=True)
        time.sleep(10)
    print('all finished, failed: %s' % (failed or 'none'))


if __name__ == '__main__':
    main()
