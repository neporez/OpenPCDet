"""
Region-based k-fold over one or several sites (Bangeojin sectors + Jangsaengpo strips, see region_utils.py).

Inputs: ImageSets/<regions>.json (one geometry, or {"sites": {...}} with a `folds` list per site = region lists),
ImageSets/<poses>.csv (sequence_id, frame_id, x, y, yaw[, site]), the base infos kriso_ulsan_infos_<infos>.pkl and the
gt database index kriso_ulsan_dbinfos_<db>.pkl. For fold f it writes
  kriso_ulsan_infos_<prefix>{f}_train.pkl       every frame, boxes in excluded(S_f) removed; info['region'] = {fold, 'train', site, exclude: S_f}
                                                -> the dataset also removes the points in excluded(S_f) (training input)
  kriso_ulsan_infos_<prefix>{f}_val.pkl         frames with >= 1 label in val_zone(S_f), full point cloud;
                                                info['region'] = {fold, 'val', site, sectors: S_f} -> only val_zone(S_f) is evaluated
  kriso_ulsan_infos_<prefix>{f}_train_eval.pkl  as many training frames as val (evenly spaced), cropped like training
  kriso_ulsan_dbinfos_<prefix>{f}_train.pkl     gt_sampling objects whose centre is not in excluded(S_f) of their site
Frames left without any box after the cut are dropped from the train split. Frames without a world pose are skipped.

python -m pcdet.datasets.kriso.make_region_folds --regions regions_v3.json --poses world_poses_v3.csv --infos real_v3 --db real_v3 --prefix regionv3_
python -m pcdet.datasets.kriso.make_region_folds --regions bangeojin_regions10.json --poses bangeojin_world_poses_v2.csv --prefix region10_   # old single-site
"""
import argparse
import copy
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from . import region_utils
from ...utils import visibility_utils

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data' / 'kriso_ulsan'
CFG = ROOT / 'tools' / 'cfgs' / 'dataset_configs' / 'kriso_dataset.yaml'


def subset(infos, n):
    idx = np.unique(np.round(np.linspace(0, len(infos) - 1, n)).astype(int))
    return [infos[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regions', default='bangeojin_regions.json')
    ap.add_argument('--poses', default='bangeojin_world_poses.csv')
    ap.add_argument('--infos', default='real', help='base infos: kriso_ulsan_infos_<infos>.pkl')
    ap.add_argument('--db', default='real', help='gt database index: kriso_ulsan_dbinfos_<db>.pkl')
    ap.add_argument('--prefix', default='region', help="info name prefix: '<prefix>{k}_train' etc.")
    args = ap.parse_args()
    lf = yaml.safe_load(open(CFG))['LABEL_FILTER']
    rj = json.load(open(DATA / 'ImageSets' / args.regions))
    regions = region_utils.load_regions(DATA / 'ImageSets' / args.regions)
    if regions.get('multi'):
        site_folds = {s: rj['sites'][s]['folds'] for s in rj['sites']}
    else:
        site_folds = {None: rj.get('folds') or [[k] for k in range(regions['n'])]}
    n_folds = len(next(iter(site_folds.values())))
    assert all(len(v) == n_folds for v in site_folds.values()), 'every site needs the same number of folds'
    poses, site_of = {}, {}
    for r in csv.DictReader(open(DATA / 'ImageSets' / args.poses)):
        key = (r['sequence_id'], r['frame_id'])
        poses[key] = [float(r['x']), float(r['y']), float(r['yaw'])]
        site_of[key] = r.get('site') if regions.get('multi') else None
    infos = [i for i in pickle.load(open(DATA / ('kriso_ulsan_infos_%s.pkl' % args.infos), 'rb')) if (i['sequence_id'], i['frame_id']) in poses]
    for i in infos:
        key = (i['sequence_id'], i['frame_id']); i['world_pose'] = poses[key]; i['site'] = site_of[key]
    sites = sorted({i['site'] for i in infos}, key=str)
    print('frames with a world pose: %d | sites %s | %d folds' % (len(infos), {s: sum(i['site'] == s for i in infos) for s in sites}, n_folds))
    db = pickle.load(open(DATA / ('kriso_ulsan_dbinfos_%s.pkl' % args.db), 'rb'))

    def rinfo(f, site, mode):
        regs = site_folds[site][f]
        d = {'fold': f, 'mode': mode, ('sectors' if mode == 'val' else 'exclude'): list(regs)}
        if site is not None:
            d['site'] = site
        return d

    for f in range(n_folds):
        train, val = [], []
        for info in infos:
            a = info['annos']; site = info['site']; g = region_utils.geom(regions, {'site': site}); regs = site_folds[site][f]
            w = region_utils.to_world(a['boxes_3d'][:, :2], info['world_pose']) if len(a['name']) else np.zeros((0, 2))
            ex = region_utils.excluded_any(g, w, regs)
            lab = visibility_utils.is_label(a['boxes_3d'], a['num_points_in_gt'], lf)
            t = copy.deepcopy(info)
            for key in ('name', 'boxes_3d', 'num_points_in_gt'):
                t['annos'][key] = a[key][~ex]
            t['region'] = rinfo(f, site, 'train')
            if len(t['annos']['name']):
                train.append(t)
            if (lab & region_utils.val_zone_any(g, w, regs)).any():
                v = copy.deepcopy(info); v['region'] = rinfo(f, site, 'val')
                val.append(v)
        train_eval = subset(train, len(val))
        for name, part in (('train', train), ('val', val), ('train_eval', train_eval)):
            pickle.dump(part, open(DATA / ('kriso_ulsan_infos_%s%d_%s.pkl' % (args.prefix, f, name)), 'wb'))
        db_k = {}
        for cls, entries in db.items():
            keep = []
            for e in entries:
                key = (e['sequence_id'], e['frame_id']); p = poses.get(key)
                if p is None:
                    continue   # no world pose
                site = site_of[key]; g = region_utils.geom(regions, {'site': site})
                w = region_utils.to_world(np.asarray(e['box3d_lidar'][None, :2]), p)
                if not region_utils.excluded_any(g, w, site_folds[site][f])[0]:
                    keep.append(e)
            db_k[cls] = keep
        pickle.dump(db_k, open(DATA / ('kriso_ulsan_dbinfos_%s%d_train.pkl' % (args.prefix, f)), 'wb'))
        tr_lab = sum(int(visibility_utils.is_label(i['annos']['boxes_3d'], i['annos']['num_points_in_gt'], lf).sum()) for i in train)
        va = {s: [0, 0] for s in sites}
        for i in val:
            g = region_utils.geom(regions, {'site': i['site']})
            z = region_utils.val_zone_any(g, region_utils.to_world(i['annos']['boxes_3d'][:, :2], i['world_pose']), site_folds[i['site']][f])
            va[i['site']][0] += 1; va[i['site']][1] += int((visibility_utils.is_label(i['annos']['boxes_3d'], i['annos']['num_points_in_gt'], lf) & z).sum())
        print('%s%d: train %d frames / %d labels | val %d frames / %d labels in the zone (%s) | train_eval %d | db %s'
              % (args.prefix, f, len(train), tr_lab, len(val), sum(v[1] for v in va.values()),
                 ', '.join('%s %d fr / %d lab' % (s, v[0], v[1]) for s, v in va.items()), len(train_eval), {c: len(v) for c, v in db_k.items()}))


if __name__ == '__main__':
    main()
