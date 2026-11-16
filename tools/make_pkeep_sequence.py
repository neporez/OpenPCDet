"""
Stochastic point dropout for an object-only simulated sequence (decided 2026-11-11): instead of deleting points, every
point gets a KEEP PROBABILITY written into the unused 4th column of the .bin, and KrisoDataset samples Bernoulli(p) at
every __getitem__ of a training frame (meta_info['point_keep_prob'] = True), so each epoch sees a different thinning.

p(point) = min(1, k_obj * cos(theta)^ALPHA * (R0 / r)^BETA)
  theta = incidence angle between the LiDAR ray and the local surface normal (k-NN PCA normal of the object's own points),
  r = range; k_obj is solved per object so that mean(p) = target keep ratio of its (length, distance) bin,
  target = real median points-per-label / sim median points-per-box in that bin (smoothed: log-linear in length per
  distance band, clipped to [MIN_KEEP, 1]). So the point COUNTS follow the real data and WHICH points vanish follows the
  geometry (grazing incidence / far first). Points outside every box (stray) get p = 1.

python make_pkeep_sequence.py --seq dist_1000_1 --out_seq dist_1000_1_thin --split sim_dist1000_thin [--alpha 1 --beta 0]
Afterwards build the infos (python -m pcdet.datasets.kriso.kriso_dataset ... --splits sim_dist1000_thin): for flagged
sequences get_infos stores the EXPECTED count (sum of p inside the box) as num_points_in_gt.
"""
import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

import _init_path  # noqa
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import box_rotation_utils, visibility_utils

ROOT = Path(__file__).resolve().parent.parent
DATA = Path('/home/rain/dataset/kriso_ulsan')
LENGTH_BINS = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 60), (60, 80), (80, 125)]
DIST_BANDS = [(50, 100), (100, 150), (150, 200)]
MIN_KEEP = 0.15
KNN = 12


def per_bin_medians(L, D, N, min_n=20):
    out = np.full((len(LENGTH_BINS), len(DIST_BANDS)), np.nan)
    for i, (lo, hi) in enumerate(LENGTH_BINS):
        for j, (dlo, dhi) in enumerate(DIST_BANDS):
            m = (L >= lo) & (L < hi) & (D >= dlo) & (D < dhi)
            if m.sum() >= min_n:
                out[i, j] = np.median(N[m])
    return out


def smooth_ratio(real_med, sim_med):
    """Target keep ratio per (length bin, distance band) = real median / sim median where both exist (>= min_n boxes);
    bins without a real or sim median are filled from a log-linear fit of the ratio over length within the band.
    (A pure fit over-smooths: the real ratios are not monotone in length, e.g. jangsaeng's 60-80 m ships are dense.)"""
    centres = np.array([np.sqrt(max(lo, 5) * hi) for lo, hi in LENGTH_BINS])
    table = np.ones((len(LENGTH_BINS), len(DIST_BANDS)))
    fits = []
    for j in range(len(DIST_BANDS)):
        r = real_med[:, j] / sim_med[:, j]; ok = np.isfinite(r) & (r > 0)
        if ok.sum() >= 3:
            b, a = np.polyfit(np.log(centres[ok]), np.log(r[ok]), 1)
        else:
            a, b = np.log(np.nanmedian(r[ok])) if ok.any() else 0.0, 0.0
        fits.append((float(a), float(b)))
        filled = np.where(ok, r, np.exp(a + b * np.log(centres)))
        table[:, j] = np.clip(filled, MIN_KEEP, 1.0)
    return table, fits


def target_for(L, r, table):
    j = int(np.clip(np.searchsorted([b[1] for b in DIST_BANDS], r), 0, len(DIST_BANDS) - 1))   # r < 50 -> band 0, > 200 -> last
    i = int(np.clip(np.searchsorted([b[1] for b in LENGTH_BINS], L), 0, len(LENGTH_BINS) - 1))
    return float(table[i, j])


def normals_cos(pts):
    """|cos(incidence)| per point from k-NN PCA normals (orientation-free)."""
    if len(pts) < KNN + 1:
        return np.ones(len(pts))
    tree = cKDTree(pts); _, idx = tree.query(pts, k=KNN + 1)
    nb = pts[idx] - pts[idx].mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', nb, nb)
    w, v = np.linalg.eigh(cov); n = v[:, :, 0]                    # eigenvector of the smallest eigenvalue
    d = pts / np.maximum(np.linalg.norm(pts, axis=1, keepdims=True), 1e-6)
    return np.abs(np.einsum('ni,ni->n', n, d))


def solve_k(g, target, iters=40):
    """k such that mean(min(1, k g)) = target (g in [0, 1]); returns p."""
    if target >= 1.0 or len(g) == 0:
        return np.ones(len(g))
    lo, hi = 0.0, 1e6
    for _ in range(iters):
        k = 0.5 * (lo + hi); m = np.minimum(1.0, k * g).mean()
        if m < target: lo = k
        else: hi = k
    return np.minimum(1.0, 0.5 * (lo + hi) * g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='dist_1000_1')
    ap.add_argument('--out_seq', default='dist_1000_1_thin')
    ap.add_argument('--split', default='sim_dist1000_thin')
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--beta', type=float, default=0.0)
    ap.add_argument('--r0', type=float, default=100.0)
    args = ap.parse_args()
    cfg_from_yaml_file(str(ROOT / 'tools/cfgs/kriso_models/kriso_voxelnext.yaml'), cfg); LF = cfg.DATA_CONFIG.LABEL_FILTER

    # real reference: points per LABEL per (length, distance) bin
    real_pkl = DATA / 'kriso_ulsan_infos_real_v4.pkl'   # 2026-11-15: dataset v4 reference (the old real.pkl is archived)
    if not real_pkl.exists():
        real_pkl = DATA / 'kriso_ulsan_infos_real.pkl'
    real = pickle.load(open(real_pkl, 'rb')); R = []
    for i in real:
        b = i['annos']['boxes_3d']; n = i['annos']['num_points_in_gt']
        if len(b):
            lab = visibility_utils.is_label(b, n, LF); R.append(np.c_[b[lab, 3], np.hypot(b[lab, 0], b[lab, 1]), n[lab]])
    R = np.concatenate(R)
    # sim: points per box, counted on the sequence itself
    src = DATA / 'data' / args.seq; J = json.load(open(src / ('%s.json' % args.seq)))
    frames = J['frames']; per_frame = []
    for fr in frames:
        pts = np.fromfile(src / 'lidar_roof' / ('%s.bin' % fr['frame_id']), dtype=np.float32).reshape(-1, 4)
        b = np.array(fr['annos']['boxes_3d'], dtype=np.float64)
        b9 = box_rotation_utils.sim_boxes_to_std(b.copy()) if b.shape[1] == 9 else np.c_[b, np.zeros((len(b), 2))]   # 9-value sim boxes (v3 captures)
        mask = box_rotation_utils.points_in_boxes_9dof(pts[:, :3].astype(np.float64), b9)   # (boxes, points)
        per_frame.append((pts, b, mask))
    S = np.concatenate([np.c_[b[:, 3], np.hypot(b[:, 0], b[:, 1]), mask.sum(1)] for _, b, mask in per_frame])
    real_med, sim_med = per_bin_medians(R[:, 0], R[:, 1], R[:, 2]), per_bin_medians(S[:, 0], S[:, 1], S[:, 2])
    table, fits = smooth_ratio(real_med, sim_med)
    print('target keep ratio (real / sim, smoothed), rows = length bins, cols = %s' % DIST_BANDS)
    for (lo, hi), row, rm, sm in zip(LENGTH_BINS, table, real_med, sim_med):
        print('  %3d-%3d m: ' % (lo, hi) + '  '.join('%.2f (real %4.0f / sim %5.0f)' % (t, a, c) for t, a, c in zip(row, rm, sm)))

    # write the thinned sequence: p in column 4
    out = DATA / 'data' / args.out_seq; (out / 'lidar_roof').mkdir(parents=True, exist_ok=True)
    rows = []; new_frames = []
    for fr, (pts, b, mask) in zip(frames, per_frame):
        p = np.ones(len(pts), dtype=np.float32); r_pt = np.linalg.norm(pts[:, :3], axis=1)
        exp_counts = []
        for bi in range(len(b)):
            m = mask[bi]
            if m.sum() == 0:
                exp_counts.append(0.0); continue
            q = pts[m, :3].astype(np.float64)
            g = normals_cos(q) ** args.alpha * (args.r0 / np.maximum(r_pt[m], 1.0)) ** args.beta
            tgt = target_for(b[bi, 3], np.hypot(b[bi, 0], b[bi, 1]), table)
            p[m] = solve_k(g, tgt); exp_counts.append(float(p[m].sum()))
            rows.append([b[bi, 3], np.hypot(b[bi, 0], b[bi, 1]), m.sum(), p[m].sum(), tgt])
        pts_out = pts.copy(); pts_out[:, 3] = p; pts_out.astype(np.float32).tofile(out / 'lidar_roof' / ('%s.bin' % fr['frame_id']))
        f2 = dict(fr); f2['sequence_id'] = args.out_seq; f2['annos'] = dict(fr['annos'], expected_points=exp_counts); new_frames.append(f2)
    meta = dict(J); meta['meta_info'] = dict(J['meta_info'], point_keep_prob=True, keep_model={'alpha': args.alpha, 'beta': args.beta, 'r0': args.r0, 'source_seq': args.seq, 'target_table': table.tolist(), 'fits': fits})
    meta['frames'] = new_frames; json.dump(meta, open(out / ('%s.json' % args.out_seq), 'w'))
    split_file = DATA / 'ImageSets' / ('%s.txt' % args.split); names = [l.strip() for l in open(split_file)] if split_file.exists() else []
    if args.out_seq not in names:
        names.append(args.out_seq); open(split_file, 'w').write('\n'.join(names) + '\n')
    A = np.array(rows)
    print('%s: %d frames, %d boxes | expected points: median %.0f (was %.0f), overall keep %.2f' % (out, len(new_frames), len(A), np.median(A[:, 3]), np.median(A[:, 2]), A[:, 3].sum() / A[:, 2].sum()))
    print('expected points per box after thinning vs real, by bin (median):')
    E = per_bin_medians(A[:, 0], A[:, 1], A[:, 3])
    for (lo, hi), e, rm in zip(LENGTH_BINS, E, real_med):
        print('  %3d-%3d m: ' % (lo, hi) + '  '.join('thin %5.0f / real %4.0f' % (x, y) for x, y in zip(e, rm)))


if __name__ == '__main__':
    main()
