"""
Re-score the saved in-training evaluations of a train_with_val.py run with the current evaluation code and add the
metrics that are not in its tensorboard yet (e.g. the boat length bins len0-40m / len40m-inf added later).

Every periodic evaluation keeps its predictions in eval/eval_during_train/epoch_<N>/<split>/result.pkl, so nothing has to
be re-run. Writes a new event file into <run>/tensorboard (tags val/... for the test split, train_eval/... for the
training subset), only for tags that the run does not have yet. Run it after the training has finished.

python rescore_run.py --run ../output/kriso_models/kriso_voxelnext/cross_site_real_ep80
python rescore_run.py --run ... --iou 0.7 --prefix iou0.7     # other IoU threshold, tags val_iou0.7/...
"""
import _init_path
import argparse
import contextlib
import copy
import io
import pickle
import re
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.kriso.eval.evaluation import get_evaluation_results
from pcdet.utils import common_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, help='run directory (with tensorboard/ and eval/eval_during_train/)')
    parser.add_argument('--iou', type=float, default=None, help='IoU threshold (default: the evaluation default, 0.5)')
    parser.add_argument('--prefix', default=None, help='tag suffix for a different IoU, e.g. iou0.7 -> val_iou0.7/...')
    args = parser.parse_args()
    run = Path(args.run).resolve()

    # dataset config of the run: the yaml copied into the run dir + the --set values logged in its train log
    cfg_from_yaml_file(str(Path(__file__).resolve().parent / 'cfgs/kriso_models/kriso_voxelnext.yaml'), cfg)
    log = sorted(run.glob('train_*.log'))[-1].read_text(errors='ignore')
    for key in ('DATA_SPLIT.test', 'EVAL_IGNORE_LENGTH_ABOVE', 'EVAL_IGNORE_NON_LABELS', 'EVAL_FILTER_GT_BY_RANGE'):
        m = re.search(r'cfg\.DATA_CONFIG\.%s: (\S+)' % re.escape(key), log)
        if m:
            val = yaml.safe_load(m.group(1))
            if '.' in key:
                cfg.DATA_CONFIG[key.split('.')[0]][key.split('.')[1]] = val
            else:
                cfg.DATA_CONFIG[key] = val
    m = re.search(r'train_eval_split\s+(\S+)', log)
    train_eval_split = m.group(1) if m and m.group(1) != 'None' else None

    acc = EventAccumulator(str(run / 'tensorboard'), size_guidance={'scalars': 0})
    acc.Reload()
    existing = set(acc.Tags().get('scalars', []))
    writer = SummaryWriter(log_dir=str(run / 'tensorboard'), filename_suffix='.rescored')
    logger = common_utils.create_logger()
    iou_thresholds = {'boat': args.iou} if args.iou is not None else None

    splits = [(cfg.DATA_CONFIG.DATA_SPLIT['test'], 'val')] + ([(train_eval_split, 'train_eval')] if train_eval_split else [])
    for split, tb in splits:
        if args.prefix:
            tb = '%s_%s' % (tb, args.prefix)
        data_cfg = copy.deepcopy(cfg.DATA_CONFIG)
        data_cfg.DATA_SPLIT['test'] = split
        dataset, _, _ = build_dataloader(data_cfg, cfg.CLASS_NAMES, 1, False, workers=0, logger=logger, training=False)
        gt = dataset._eval_gt_annos()
        epochs = sorted(int(p.parent.parent.name.split('_')[1]) for p in run.glob('eval/eval_during_train/epoch_*/%s/result.pkl' % split))
        added = 0
        for ep in epochs:
            det = pickle.load(open(run / ('eval/eval_during_train/epoch_%d/%s/result.pkl' % (ep, split)), 'rb'))
            with contextlib.redirect_stdout(io.StringIO()):
                _, ret = get_evaluation_results(gt, [dict(d) for d in det], cfg.CLASS_NAMES, iou_thresholds=iou_thresholds)
            for key, val in ret.items():
                tag = '%s/%s' % (tb, key)
                if tag in existing or not np.isfinite(val):
                    continue
                writer.add_scalar(tag, float(val), ep)
                added += 1
        print('%s (%s): %d epochs re-scored, %d values added' % (split, tb, len(epochs), added))
    writer.close()


if __name__ == '__main__':
    main()
