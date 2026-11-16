"""
Summarise a k-fold grid from run_kfold.py: validation AP of every fold and mean +- std over the folds per setting.

    python kfold_summary.py --group dropout                 # last epoch (the trained model)
    python kfold_summary.py --group dropout --epoch 80      # an intermediate evaluation (not a shorter schedule!)

Reads the in-training evaluations from each run's tensorboard (val/AP_3D_mean/..., val/AP_BEV_mean/..., val/loss, and
train_eval/AP_3D_mean/overall: gap = train AP - val AP on the fold's training subset).
std is the sample std over the folds (ddof = 1). A fold that has not reached the epoch is left out (n says how many).
Also writes <group>/summary.csv with one row per run.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OUT = Path(__file__).resolve().parent.parent / 'output' / 'kriso_models' / 'kriso_voxelnext'
BINS = ['overall', '0-100m', '100-150m', '0-150m', '150m-inf', 'len0-40m', 'len40m-inf']


def read_run(run_dir):
    acc = EventAccumulator(str(run_dir / 'tensorboard'), size_guidance={'scalars': 0})
    acc.Reload()
    tags = acc.Tags().get('scalars', [])
    series = {}
    for mode in ('3D', 'BEV'):
        for b in BINS:
            tag = 'val/AP_%s_mean/%s' % (mode, b)
            if tag in tags:
                series['AP%s %s' % (mode, b)] = {e.step: e.value for e in acc.Scalars(tag)}
    if 'val/loss' in tags:
        series['val loss'] = {e.step: e.value for e in acc.Scalars('val/loss')}
    tag = 'train_eval/AP_3D_mean/overall'
    if tag in tags:
        series['train AP3D overall'] = {e.step: e.value for e in acc.Scalars(tag)}
        series['gap AP3D overall'] = {k: v - series['AP3D overall'].get(k, np.nan) for k, v in series['train AP3D overall'].items()}
    return series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True)
    parser.add_argument('--epoch', type=int, default=None, help='default: the last evaluated epoch of each run')
    args = parser.parse_args()

    group_dir = OUT / args.group
    rows, per_setting = [], defaultdict(list)
    for run_dir in sorted(group_dir.glob('*_fold*')):
        m = re.match(r'(.+)_fold(\d+)$', run_dir.name)
        if not m or not (run_dir / 'tensorboard').exists():
            continue
        series = read_run(run_dir)
        if 'AP3D overall' not in series:
            continue
        steps = sorted(series['AP3D overall'])
        epoch = args.epoch if args.epoch is not None else steps[-1]
        if epoch not in series['AP3D overall']:
            continue
        row = {'setting': m.group(1), 'fold': int(m.group(2)), 'epoch': epoch, 'done': (run_dir / 'done').exists()}
        for key, s in series.items():
            row[key] = s.get(epoch, np.nan)
        if 'val loss' in series:
            best = min(series['val loss'], key=series['val loss'].get)
            row['min val loss epoch'] = best
        rows.append(row)
        per_setting[m.group(1)].append(row)
    if not rows:
        print('no evaluated runs in', group_dir)
        return

    keys = ['AP3D overall', 'AP3D 0-100m', 'AP3D 100-150m', 'AP3D 150m-inf', 'APBEV overall', 'train AP3D overall',
            'gap AP3D overall', 'val loss']
    print('per run (epoch %s):' % ('last' if args.epoch is None else args.epoch))
    print('  %-12s %4s %5s %4s  ' % ('setting', 'fold', 'epoch', 'done') + '  '.join('%13s' % k for k in keys))
    for r in sorted(rows, key=lambda r: (r['setting'], r['fold'])):
        print('  %-12s %4d %5d %4s  ' % (r['setting'], r['fold'], r['epoch'], 'y' if r['done'] else 'n')
              + '  '.join('%13.2f' % r.get(k, np.nan) if k != 'val loss' else '%13.4f' % r.get(k, np.nan) for k in keys))
    print('\nmean +- std over folds:')
    print('  %-12s %2s  ' % ('setting', 'n') + '  '.join('%15s' % k for k in keys))
    for setting, rs in sorted(per_setting.items()):
        cells = []
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rs], dtype=float)
            v = v[np.isfinite(v)]
            sd = v.std(ddof=1) if len(v) > 1 else np.nan
            cells.append(('%7.2f +- %5.2f' if k != 'val loss' else '%7.4f +-%6.4f') % (v.mean(), sd))
        print('  %-12s %2d  ' % (setting, len(rs)) + '  '.join('%15s' % c for c in cells))

    fields = ['setting', 'fold', 'epoch', 'done'] + sorted({k for r in rows for k in r} - {'setting', 'fold', 'epoch', 'done'})
    with open(group_dir / 'summary.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['setting'], r['fold'])))
    print('\nwritten', group_dir / 'summary.csv')


if __name__ == '__main__':
    main()
