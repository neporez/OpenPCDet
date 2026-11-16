"""
Interactive Open3D viewer for detection results: one or more checkpoints on the frames of a split, GT and predictions
overlaid, step through the frames and switch between the checkpoints on the same frame.

colours (default): green = prediction, red = GT label up to --big_length (35 m), blue = GT label longer than that;
          GT that is not a label (don't care) is not drawn.  With --detailed_colors:
          Region-fold splits (regionv3_k_val + --set DATA_CONFIG.REGION_FILE regions_v3.json): only the fold's evaluated zone
          counts (labels outside it = don't care, predictions outside it dropped, as in the evaluation); points of that
          UNSEEN zone are drawn white, the rest (seen in training) dark grey.
          red = GT label, magenta = GT label longer than --big_length (35 m), orange = GT that is not a label
          (don't care, LABEL_FILTER)
          green = prediction matching a GT label (IoU >= --iou), yellow = prediction only matching a don't-care GT,
          pink = unmatched prediction whose centre lies inside a big ship (fragment of it),
          blue = other prediction without a match (false positive); white points; the axes mark the sensor
keys:     N / Right = next frame, B / Left = previous frame, M = next checkpoint, G = GT on / off,
          P = predictions on / off, space = autoplay on / off, V = save the camera to --view_file, L = load it, Q / Esc = quit
The terminal prints, per frame and checkpoint, TP / FP / FN (score >= --score) and the boxes that were missed.

python vis_detection.py \\
    --ckpt ../output/kriso_models/kriso_voxelnext/fold0_ep80/ckpt/checkpoint_best.pth \\
           ../output/kriso_models/kriso_voxelnext/cross_site_real_ep80/ckpt/checkpoint_best.pth \\
    --names fold0 cross_site --split cross_site_val --scene scene_006 --min_labels 4
"""
import _init_path
import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import open3d
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.datasets.kriso.eval import box_iou
from pcdet.models import build_network, load_data_to_gpu
from pcdet.ops.iou3d_nms import iou3d_nms_utils
from pcdet.utils import box_rotation_utils, common_utils, visibility_utils

ROOT = Path(__file__).resolve().parent.parent
EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
C_GT, C_GT_IGN = (1.0, 0.1, 0.1), (1.0, 0.6, 0.0)
C_TP, C_IGN, C_FP = (0.1, 1.0, 0.1), (1.0, 1.0, 0.0), (0.2, 0.6, 1.0)
C_BIG, C_FRAG = (1.0, 0.0, 1.0), (1.0, 0.55, 0.8)


def boxes_mesh(boxes, colors, radius):
    """all box edges (+ a heading line) of all boxes as one triangle mesh of thin cylinders"""
    mesh = open3d.geometry.TriangleMesh()
    if len(boxes) == 0:
        return mesh
    corners = box_rotation_utils.boxes_to_corners_9dof(boxes)
    rot = box_rotation_utils.boxes_to_rotation_matrices(boxes)
    for b, cor, R, col in zip(boxes, corners, rot, colors):
        segs = [(cor[i], cor[j]) for i, j in EDGES] + [(b[:3], b[:3] + R @ np.array([b[3] / 2, 0.0, 0.0]))]
        for a, c in segs:
            vec = c - a
            length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            cyl = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=6)
            d = vec / length
            axis = np.cross([0, 0, 1], d)
            if np.linalg.norm(axis) > 1e-8:
                cyl.rotate(open3d.geometry.get_rotation_matrix_from_axis_angle(
                    axis / np.linalg.norm(axis) * np.arccos(np.clip(d[2], -1, 1))), center=(0, 0, 0))
            elif d[2] < 0:
                cyl.rotate(open3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))
            cyl.translate((a + c) / 2)
            cyl.paint_uniform_color(col)
            mesh += cyl
    mesh.compute_vertex_normals()
    return mesh


def iou_matrix(g, p, mode):
    if len(g) == 0 or len(p) == 0:
        return np.zeros((len(g), len(p)))
    if mode == 'bev':
        return iou3d_nms_utils.boxes_bev_iou_cpu(torch.tensor(g[:, :7]).float(), torch.tensor(p[:, :7]).float()).numpy()
    return box_iou.iou3d_matrix(g, p)


def match(g, lab, p, score, iou_th, mode):
    """greedy by score: each prediction takes the best free GT. returns (pred status: tp / ign / fp, gt matched)"""
    iou = iou_matrix(g, p, mode)
    used = np.zeros(len(g), bool)
    status = np.array(['fp'] * len(p), dtype=object)
    for j in np.argsort(-score):
        if len(g) == 0:
            break
        cand = np.where(~used & (iou[:, j] >= iou_th))[0]
        if len(cand) == 0:
            continue
        i = cand[np.argmax(iou[cand, j])]
        used[i] = True
        status[j] = 'tp' if lab[i] else 'ign'
    return status, used


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', default=str(ROOT / 'tools/cfgs/kriso_models/kriso_voxelnext.yaml'))
    parser.add_argument('--ckpt', nargs='+', required=True)
    parser.add_argument('--names', nargs='+', default=None, help='a short name per checkpoint')
    parser.add_argument('--split', default='cross_site_val', help='INFO_PATH name of the frames to show')
    parser.add_argument('--sequence', default=None, help='only this sequence (ulsan_0609 / ulsan_1105 / yaw_1000_1 ...)')
    parser.add_argument('--bag', default=None, help='bag name suffix: clockwise / counterclockwise / jangsaeng (sequences with a mapping csv)')
    parser.add_argument('--scene', default=None, help='scene_000 ... (sequences with a <seq>_mapping.csv)')
    parser.add_argument('--site', default=None, help='bangeojin / jangsaengpo (ImageSets/world_poses_v3.csv)')
    parser.add_argument('--frames', nargs='+', default=None, help='explicit frames, e.g. ulsan_1105:001480')
    parser.add_argument('--min_labels', type=int, default=0, help='only frames with at least this many GT labels')
    parser.add_argument('--big_length', type=float, default=35.0, help='GT labels longer than this (m) are drawn magenta')
    parser.add_argument('--detailed_colors', action='store_true', help='colour predictions by match status (TP / FP / fragment ...)')
    parser.add_argument('--only_big', action='store_true', help='only frames with at least one GT label longer than --big_length')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--score', type=float, default=0.3, help='predictions below this score are not drawn')
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--iou_mode', choices=['3d', 'bev'], default='3d', help='3d = exact 6-DoF IoU (as AP_3D)')
    parser.add_argument('--interval', type=float, default=1.0, help='seconds per frame in autoplay (space)')
    parser.add_argument('--point_size', type=float, default=2.0)
    parser.add_argument('--line_radius', type=float, default=0.15)
    parser.add_argument('--view_file', default=str(ROOT / 'output' / 'vis_view.json'))
    parser.add_argument('--max_frames', type=int, default=None, help='close after showing this many frames (testing)')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg_from_list(['DATA_CONFIG.DATA_SPLIT.test', args.split], cfg)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, cfg)
    logger = common_utils.create_logger()
    ds, _, _ = build_dataloader(cfg.DATA_CONFIG, cfg.CLASS_NAMES, 1, False, workers=0, logger=logger, training=False)
    label_filter = cfg.DATA_CONFIG.get('LABEL_FILTER', None)

    # <seq>_mapping.csv (bag / scene per frame) of every real sequence that has one (ulsan_1105_1106, old ulsan_1105 ...),
    # and the site (bangeojin / jangsaengpo) per frame from ImageSets/world_poses_v3.csv
    mapping = {}
    for mp in (ROOT / 'data/kriso_ulsan/data').glob('*/*_mapping.csv'):
        seq = mp.parent.name
        for r in csv.DictReader(open(mp)):
            mapping[(seq, r['frame_id'])] = r
    site_of = {}
    wp = ROOT / 'data/kriso_ulsan/ImageSets/world_poses_v3.csv'
    if wp.exists():
        site_of = {(r['sequence_id'], r['frame_id']): r.get('site', '') for r in csv.DictReader(open(wp))}

    def where(info):
        r = mapping.get((info['sequence_id'], info['frame_id']))
        if r:
            return '%s/%s' % (r['bag'].split('_', 2)[2], r['scene'])
        return info['sequence_id']

    def num_labels(info):
        a = info['annos']
        return int(visibility_utils.is_label(a['boxes_3d'], a['num_points_in_gt'], label_filter).sum()) if label_filter else len(a['name'])

    wanted = {tuple(f.split(':')) for f in args.frames} if args.frames else None
    frames = []
    for k, info in enumerate(ds.once_infos):
        key = (info['sequence_id'], info['frame_id'])
        if wanted is not None and key not in wanted:
            continue
        if args.sequence and info['sequence_id'] != args.sequence:
            continue
        if (args.bag or args.scene) and key not in mapping:
            continue
        if args.bag and not mapping[key]['bag'].endswith(args.bag):
            continue
        if args.scene and mapping[key]['scene'] != args.scene:
            continue
        if args.site and site_of.get(key, '') != args.site:
            continue
        if num_labels(info) < args.min_labels:
            continue
        if args.only_big:
            a = info['annos']
            lab_ = visibility_utils.is_label(a['boxes_3d'], a['num_points_in_gt'], label_filter) if label_filter else np.ones(len(a['name']), bool)
            if not (lab_ & (a['boxes_3d'][:, 3] > args.big_length)).any():
                continue
        frames.append(k)
    print('%d frames of %s to show' % (len(frames), args.split))
    if not frames:
        return

    names = args.names if args.names and len(args.names) == len(args.ckpt) else [Path(c).parent.parent.name + '/' + Path(c).stem for c in args.ckpt]
    models = []
    for ckpt in args.ckpt:
        m = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), ds)
        m.load_params_from_file(ckpt, logger=logger, to_cpu=False)
        models.append(m.cuda().eval())
    cache = {}

    def predict(k, mi):
        if (k, mi) not in cache:
            batch = ds.collate_batch([ds[k]])
            load_data_to_gpu(batch)
            with torch.no_grad():
                pred, _ = models[mi](batch)
            b = pred[0]['pred_boxes'].cpu().numpy().astype(np.float64)
            s = pred[0]['pred_scores'].cpu().numpy()
            keep = s >= args.score
            cache[(k, mi)] = (b[keep], s[keep])
        return cache[(k, mi)]

    state = {'i': max(0, min(args.start, len(frames) - 1)), 'm': 0, 'gt': True, 'pred': True,
             'play': False, 'last': 0.0, 'first': True, 'shown': 0}
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='detections (N/B frame, M checkpoint, G gt, P pred, V/L camera, Q quit)')
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])

    def show():
        k, mi = frames[state['i']], state['m']
        info = ds.once_infos[k]
        g = info['annos']['boxes_3d'].astype(np.float64)
        lab = visibility_utils.is_label(g, info['annos']['num_points_in_gt'], label_filter) if label_filter else np.ones(len(g), bool)
        p, s = predict(k, mi)
        zone_pts = None
        if 'region' in info and getattr(ds, 'regions', None) is not None:   # region fold: only the fold's evaluated zone counts (as in the evaluation)
            zone_g = ds._region_zone(info, g) if len(g) else np.zeros(0, bool)
            lab = lab & zone_g
            if len(p):
                keep = ds._region_zone(info, p); p, s = p[keep], s[keep]
            from pcdet.datasets.kriso import region_utils
        status, gt_used = match(g, lab, p, s, args.iou, args.iou_mode)
        big = lab & (g[:, 3] > args.big_length)
        if big.any() and len(p):  # unmatched predictions inside a big ship (2 m margin) = fragments of it
            b2 = g[big].copy(); b2[:, 3:6] += 4.0
            inside = box_rotation_utils.points_in_boxes_9dof(p[:, :3], b2).any(axis=0)
            status = np.where((status == 'fp') & inside, 'frag', status)
        pts = ds.get_lidar(info['sequence_id'], info['frame_id'])[:, :3]
        pts = pts[np.isfinite(pts).all(axis=1)]

        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pts.astype(np.float64)))
        pcd.paint_uniform_color([0.85, 0.85, 0.85])
        if 'region' in info and getattr(ds, 'regions', None) is not None:    # points of the evaluated (unseen) zone white, the rest dark grey
            from pcdet.datasets.kriso import region_utils
            w = region_utils.to_world(pts[:, :2], info['world_pose'])
            inz = region_utils.zone_mask(ds.regions, w, info['region']['fold'], info['region']['mode'], info['region'])
            col = np.where(inz[:, None], [[0.95, 0.95, 0.95]], [[0.35, 0.35, 0.4]]); pcd.colors = open3d.utility.Vector3dVector(col)
        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=state['first'])
        vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0), reset_bounding_box=False)
        if state['gt'] and len(g):
            if args.detailed_colors:
                vis.add_geometry(boxes_mesh(g, [(C_BIG if bg else C_GT) if l else C_GT_IGN for l, bg in zip(lab, big)], args.line_radius),
                                 reset_bounding_box=False)
            elif lab.any():  # simple: labels only, red up to big_length, blue above
                vis.add_geometry(boxes_mesh(g[lab], [(0.2, 0.4, 1.0) if bg else (1.0, 0.1, 0.1) for bg in big[lab]], args.line_radius),
                                 reset_bounding_box=False)
        if state['pred'] and len(p):
            col = {'tp': C_TP, 'ign': C_IGN, 'fp': C_FP, 'frag': C_FRAG}
            pcol = [col[x] for x in status] if args.detailed_colors else [(0.1, 1.0, 0.1)] * len(p)
            vis.add_geometry(boxes_mesh(p, pcol, args.line_radius * 0.8), reset_bounding_box=False)
        if state['first'] and os.path.exists(args.view_file):
            vis.get_view_control().convert_from_pinhole_camera_parameters(
                open3d.io.read_pinhole_camera_parameters(args.view_file), allow_arbitrary=True)
        state['first'] = False

        tp = int((status == 'tp').sum()); fp = int((status == 'fp').sum()); frag = int((status == 'frag').sum())
        fn = int((lab & ~gt_used).sum())
        print('[%d/%d] %s %s (%s) | %s | labels %d (> %.0f m: %d, found %d): TP %d  FP %d (+ %d inside big ships)  FN %d'
              '  (score >= %.2f, %s IoU >= %.2f)'
              % (state['i'] + 1, len(frames), info['sequence_id'], info['frame_id'], where(info), names[mi],
                 lab.sum(), args.big_length, big.sum(), (big & gt_used).sum(), tp, fp, frag, fn, args.score, args.iou_mode.upper(), args.iou))
        for b in g[lab & ~gt_used]:
            print('      missed: length %5.1f m  width %4.1f m  dist %5.1f m%s' % (b[3], b[4], np.linalg.norm(b[:3]), '  (big)' if b[3] > args.big_length else ''))
        state['shown'] += 1

    def step(delta):
        def cb(v):
            state['i'] = (state['i'] + delta) % len(frames)
            show(); state['last'] = time.time()
            return True
        return cb

    def next_model(v):
        state['m'] = (state['m'] + 1) % len(models)
        show()
        return True

    def toggle(key):
        def cb(v):
            state[key] = not state[key]
            if key == 'play':
                print('autoplay' if state['play'] else 'paused')
                return False
            show()
            return True
        return cb

    def save_view(v):
        open3d.io.write_pinhole_camera_parameters(args.view_file, v.get_view_control().convert_to_pinhole_camera_parameters())
        print('camera saved to', args.view_file)
        return False

    def load_view(v):
        if os.path.exists(args.view_file):
            v.get_view_control().convert_from_pinhole_camera_parameters(
                open3d.io.read_pinhole_camera_parameters(args.view_file), allow_arbitrary=True)
        return False

    def tick(v):
        if args.max_frames is not None and state['shown'] >= args.max_frames:
            v.close()
            return False
        if state['play'] and time.time() - state['last'] >= args.interval:
            state['i'] = (state['i'] + 1) % len(frames)
            show(); state['last'] = time.time()
            return True
        return False

    for key, cb in ((ord('N'), step(1)), (262, step(1)), (ord('B'), step(-1)), (263, step(-1)), (ord('M'), next_model),
                    (ord('G'), toggle('gt')), (ord('P'), toggle('pred')), (ord(' '), toggle('play')),
                    (ord('V'), save_view), (ord('L'), load_view)):
        vis.register_key_callback(key, cb)
    vis.register_animation_callback(tick)
    show()
    state['last'] = time.time()
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
