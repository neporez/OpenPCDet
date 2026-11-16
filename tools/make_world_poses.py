"""
World (reference-frame) ego poses per dataset frame for the region folds: ImageSets/world_poses_<tag>.csv
    sequence_id, frame_id, x, y, yaw, site, bag, scene
from the kriso_world scene poses (scene_XXX/poses.csv, refined) + scene.json origin_in_reference_frame + site tag, joined to the
dataset frames through each sequence's <seq>_mapping.csv (export_dataset.py). Jangsaengpo scenes get the label-ICP alignment of
ImageSets/jangsaengpo_scene_align.json (as world_poses_v3.csv did); Bangeojin scenes are used raw (reference frame).
python make_world_poses.py --seqs ulsan_1105_1106 ulsan_0609_world --out world_poses_v4.csv [--check world_poses_v3.csv]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

W = Path('/media/rain/Extreme SSD/kriso_world')
D = Path(__file__).resolve().parent.parent / 'data' / 'kriso_ulsan'


def q_yaw(qx, qy, qz, qw):
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seqs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--check', default=None, help='existing csv to compare common frames against')
    args = ap.parse_args()
    AL = json.load(open(D / 'ImageSets' / 'jangsaengpo_scene_align.json'))['scenes']
    rows = []; cache = {}
    for seq in args.seqs:
        for m in csv.DictReader(open(D / 'data' / seq / ('%s_mapping.csv' % seq))):
            key = (m['bag'], m['scene'])
            if key not in cache:
                sd = W / m['bag'] / m['scene']; meta = json.load(open(sd / 'scene.json'))
                poses = {r['seq']: r for r in csv.DictReader(open(sd / 'poses.csv'))}
                cache[key] = (meta, poses)
            meta, poses = cache[key]; p = poses[m['seq']]
            o = np.array(meta['origin_in_reference_frame'][:2]); xy = np.array([float(p['x']), float(p['y'])]) + o
            yaw = q_yaw(*(float(p[k]) for k in ('qx', 'qy', 'qz', 'qw')))
            site = meta.get('site')
            a = AL.get('%s/%s' % (m['bag'], m['scene']))
            if site == 'jangsaengpo' and a:
                R = np.array(a['R']); xy = R @ xy + np.array(a['t']); yaw += float(np.arctan2(R[1, 0], R[0, 0]))
            rows.append([seq, m['frame_id'], '%.3f' % xy[0], '%.3f' % xy[1], '%.6f' % yaw, site, m['bag'], m['scene']])
    out = D / 'ImageSets' / args.out
    with open(out, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sequence_id', 'frame_id', 'x', 'y', 'yaw', 'site', 'bag', 'scene']); w.writerows(rows)
    from collections import Counter
    print('%s: %d frames | sites %s | sequences %s' % (out, len(rows), dict(Counter(r[5] for r in rows)), dict(Counter(r[0] for r in rows))))
    if args.check:
        old = {(r['sequence_id'], r['frame_id']): r for r in csv.DictReader(open(D / 'ImageSets' / args.check))}
        d = [abs(float(r[2]) - float(old[(r[0], r[1])]['x'])) + abs(float(r[3]) - float(old[(r[0], r[1])]['y'])) for r in rows if (r[0], r[1]) in old]
        print('check vs %s: %d common frames, max |dx|+|dy| = %.4f m' % (args.check, len(d), max(d) if d else -1))


if __name__ == '__main__':
    main()
