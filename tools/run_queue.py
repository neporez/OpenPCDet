"""
Generic overnight queue for train_with_val.py runs: N at a time, done markers, restart-safe.

Queue file: JSON list of {"name", "train", "test", "train_eval", "epochs", "dropout" (opt), "seed" (opt), "set" (opt list),
"cfg" (opt model yaml, default kriso_voxelnext.yaml), "out" (opt output group, default "region"), "eval_interval" (opt, 0 = none)}.
Each run -> output/kriso_models/kriso_voxelnext/region/<name>/ (run.out, done marker). Every run gets
OMP_NUM_THREADS=MKL_NUM_THREADS=2. A run whose `done` exists is skipped (restart the driver after an interruption).

python run_queue.py --queue queue.json --parallel 3
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
OUT = TOOLS.parent / 'output' / 'kriso_models' / 'kriso_voxelnext' / 'region'
CFG = 'cfgs/kriso_models/kriso_voxelnext.yaml'


def log(msg):
    print('%s %s' % (time.strftime('%m-%d %H:%M:%S'), msg), flush=True)


def cmd_for(j, workers=4):
    c = [sys.executable, 'train_with_val.py', '--cfg_file', j.get('cfg', CFG), '--epochs', str(j['epochs']),
         '--extra_tag', j.get('out', 'region') + '/' + j['name'],
         '--workers', str(workers), '--seed', str(j.get('seed', 666)), '--eval_interval', str(j.get('eval_interval', 2))]
    if j.get('eval_interval', 2) > 0:
        c += ['--save_best_last', '--train_eval_split', j['train_eval']]
    c += [
         '--set', 'DATA_CONFIG.DATA_SPLIT.train', j['train'], 'DATA_CONFIG.DATA_SPLIT.test', j['test'],
         'MODEL.DENSE_HEAD.DROPOUT', repr(float(j.get('dropout', 0.0)))]
    return c + j.get('set', [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--queue', required=True)
    ap.add_argument('--parallel', type=int, default=3)
    ap.add_argument('--workers', type=int, default=4, help='dataloader workers per run (use 2 on 6-core machines)')
    ap.add_argument('--gpus', default=None, help='comma list of GPU ids to spread the parallel runs over, e.g. 0,2')
    args = ap.parse_args()
    queue = json.load(open(args.queue))
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    running, failed = [], []
    log('%d runs queued, %d at a time' % (len(queue), args.parallel))
    while queue or running:
        while queue and len(running) < args.parallel:
            j = queue.pop(0); d = OUT.parent / j.get('out', 'region') / j['name']
            if (d / 'done').exists():
                log('skip (done) ' + j['name']); continue
            d.mkdir(parents=True, exist_ok=True)
            out = open(d / 'run.out', 'w')
            run_env = dict(env)
            if args.gpus:
                gpu_list = args.gpus.split(','); run_env['CUDA_VISIBLE_DEVICES'] = gpu_list[len(running) % len(gpu_list)]
            p = subprocess.Popen(cmd_for(j, args.workers), cwd=TOOLS, stdout=out, stderr=subprocess.STDOUT, env=run_env)
            running.append((j, p, out, time.time(), d))
            log('start %s (%s -> %s, %d ep, dropout %s, seed %s)' % (j['name'], j['train'], j['test'], j['epochs'], j.get('dropout', 0.0), j.get('seed', 666)))
            time.sleep(60)
        for item in list(running):
            j, p, out, t0, d = item
            if p.poll() is None:
                continue
            out.close(); running.remove(item)
            ok = p.returncode == 0 and ((d / 'ckpt' / 'checkpoint_last.pth').exists() or (d / 'ckpt' / ('checkpoint_epoch_%d.pth' % j['epochs'])).exists())
            if ok:
                (d / 'done').write_text(time.strftime('%Y-%m-%d %H:%M:%S\n'))
            else:
                failed.append(j['name'])
            log('%s %s (%.2f h)' % ('done  ' if ok else 'FAILED', j['name'], (time.time() - t0) / 3600))
        time.sleep(30)
    log('all finished, failed: %s' % (failed or 'none'))


if __name__ == '__main__':
    main()
