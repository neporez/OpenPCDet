"""
Training on a subset of the Bangeojin regions (data-efficiency experiments): val = region V (the usual region{V}_val),
training = only the regions T (all other regions + their buffers are cut out of the training frames).

Writes (from kriso_ulsan_infos_region{V}_train.pkl, i.e. the Bangeojin frames with world poses):
  kriso_ulsan_infos_region{V}_train_r{T}.pkl        info['region'] = {fold V, 'train', exclude = all regions not in T}
  kriso_ulsan_infos_region{V}_train_r{T}_eval.pkl   as many of those frames as region{V}_val (train AP, training zone only)
  kriso_ulsan_dbinfos_region{V}_train_r{T}[_yaw1000|_yaw3000|_yaw5000].pkl   gt_sampling objects of the training zone

python -m pcdet.datasets.kriso.make_region_subset --val 2 --train 0 1
"""
import argparse
import copy
import csv
import pickle
from pathlib import Path

import numpy as np

from . import region_utils
from .make_region_folds import subset

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data' / 'kriso_ulsan'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', type=int, required=True)
    ap.add_argument('--train', type=int, nargs='+', required=True)
    args = ap.parse_args()
    regions = region_utils.load_regions(DATA / 'ImageSets' / 'bangeojin_regions.json')
    K = len(regions['boundaries'])
    assert args.val not in args.train
    out = [k for k in range(K) if k not in args.train]          # cut out: the val region and the unused ones
    tag = 'region%d_train_r%s' % (args.val, ''.join(str(t) for t in sorted(args.train)))
    base = pickle.load(open(DATA / ('kriso_ulsan_infos_region%d_train.pkl' % args.val), 'rb'))
    infos = []
    for info in base:
        a = info['annos']
        w = region_utils.to_world(a['boxes_3d'][:, :2], info['world_pose']) if len(a['name']) else np.zeros((0, 2))
        ex = region_utils.excluded_any(regions, w, out)
        t = copy.deepcopy(info)
        for key in ('name', 'boxes_3d', 'num_points_in_gt'):
            t['annos'][key] = a[key][~ex]
        t['region'] = {'fold': args.val, 'mode': 'train', 'exclude': out}
        if len(t['annos']['name']):
            infos.append(t)
    n_val = len(pickle.load(open(DATA / ('kriso_ulsan_infos_region%d_val.pkl' % args.val), 'rb')))
    pickle.dump(infos, open(DATA / ('kriso_ulsan_infos_%s.pkl' % tag), 'wb'))
    pickle.dump(subset(infos, n_val), open(DATA / ('kriso_ulsan_infos_%s_eval.pkl' % tag), 'wb'))
    poses = {(r['sequence_id'], r['frame_id']): [float(r['x']), float(r['y']), float(r['yaw'])]
             for r in csv.DictReader(open(DATA / 'ImageSets' / 'bangeojin_world_poses.csv'))}
    db = pickle.load(open(DATA / 'kriso_ulsan_dbinfos_real.pkl', 'rb'))
    db_t = {}
    for cls, entries in db.items():
        keep = []
        for e in entries:
            p = poses.get((e['sequence_id'], e['frame_id']))
            if p is not None and not region_utils.excluded_any(regions, region_utils.to_world(np.asarray(e['box3d_lidar'][None, :2]), p), out)[0]:
                keep.append(e)
        db_t[cls] = keep
    for suffix in ('', '_yaw1000', '_yaw3000', '_yaw5000'):
        pickle.dump(db_t, open(DATA / ('kriso_ulsan_dbinfos_%s%s.pkl' % (tag, suffix)), 'wb'))
    nb = sum(len(i['annos']['name']) for i in infos)
    print('%s: %d frames, %d boxes (cut out regions %s) | eval subset %d | db %s' % (tag, len(infos), nb, out, min(n_val, len(infos)),
                                                                                   {c: len(v) for c, v in db_t.items()}))


if __name__ == '__main__':
    main()
