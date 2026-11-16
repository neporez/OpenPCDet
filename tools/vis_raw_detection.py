"""
Open3D viewer for detections on RAW frames (no labels): a folder of .bin files (x, y, z, intensity float32), one checkpoint.

keys: N / Right = next frame, B / Left = previous, + / - = score threshold up / down (0.05 steps), space = autoplay,
      T = top (bird's-eye) view, V / L = save / load the camera (--view_file), Q / Esc = quit.
Boxes are green (edge radius --line_radius, 0.35 m) and the points inside a detected box are coloured green too, so a
detection stays visible even when the box edges are thinner than a pixel; the terminal prints scores per frame.

python vis_raw_detection.py --data_path <folder of .bin> --ckpt <checkpoint> [--score 0.3]
"""
import _init_path  # noqa
import argparse
import os
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from demo import DemoDataset
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import box_rotation_utils, common_utils

ROOT = Path(__file__).resolve().parent.parent
EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


def boxes_mesh(boxes, color, radius):
    mesh = o3d.geometry.TriangleMesh()
    if len(boxes) == 0:
        return mesh
    corners = box_rotation_utils.boxes_to_corners_9dof(boxes); rot = box_rotation_utils.boxes_to_rotation_matrices(boxes)
    for b, cor, R in zip(boxes, corners, rot):
        for a, c in [(cor[i], cor[j]) for i, j in EDGES] + [(b[:3], b[:3] + R @ np.array([b[3] / 2, 0.0, 0.0]))]:
            vec = c - a; length = np.linalg.norm(vec)
            if length < 1e-6:
                continue
            cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=6); d = vec / length
            axis = np.cross([0, 0, 1], d)
            if np.linalg.norm(axis) > 1e-8:
                cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * np.arccos(np.clip(d[2], -1, 1))), center=(0, 0, 0))
            elif d[2] < 0:
                cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))
            cyl.translate((a + c) / 2); cyl.paint_uniform_color(color); mesh += cyl
    mesh.compute_vertex_normals()
    return mesh


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', default=str(ROOT / 'tools/cfgs/kriso_models/kriso_voxelnext.yaml'))
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--score', type=float, default=0.3)
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--point_size', type=float, default=2.0)
    ap.add_argument('--line_radius', type=float, default=0.35, help='box edge radius in metres (0.15 is sub-pixel at the default zoom of a 200 m scene)')
    ap.add_argument('--no_highlight', action='store_true', help='do not colour the points inside the detected boxes green')
    ap.add_argument('--view_file', default=str(ROOT / 'output' / 'vis_view.json'))
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                    help='e.g. --set MODEL.DENSE_HEAD.DROPOUT 0.25 : MUST match the checkpoint (dropout shifts the head layer indices)')
    args = ap.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs:
        cfg_from_list(args.set_cfgs, cfg)
    logger = common_utils.create_logger()
    ds = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=Path(args.data_path), logger=logger)
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=ds)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False); model.cuda().eval()
    print('%d frames, checkpoint %s' % (len(ds), args.ckpt), flush=True)
    cache = {}

    def predict(i):
        if i not in cache:
            batch = ds.collate_batch([ds[i]]); load_data_to_gpu(batch)
            with torch.no_grad():
                pred, _ = model(batch)
            cache[i] = (pred[0]['pred_boxes'].cpu().numpy().astype(np.float64), pred[0]['pred_scores'].cpu().numpy())
        return cache[i]

    st = {'i': max(0, min(args.start, len(ds) - 1)), 'score': args.score, 'play': False, 'last': 0.0, 'first': True}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="raw detections (N/B frame, +/- score, space play, T top view, V/L camera, Q quit)")
    opt = vis.get_render_option(); opt.point_size = args.point_size; opt.background_color = np.array([0.05, 0.05, 0.05])

    def show():
        i = st['i']; b, s = predict(i); keep = s >= st['score']
        pts = np.fromfile(ds.sample_file_list[i], dtype=np.float32).reshape(-1, 4)[:, :3]; pts = pts[np.isfinite(pts).all(1)]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        col = np.full((len(pts), 3), 0.85); n_in = 0
        if keep.any() and not args.no_highlight:  # points inside a detected box -> green (visible at any zoom, unlike thin box edges)
            inside = box_rotation_utils.points_in_boxes_9dof(pts.astype(np.float64), b[keep]).any(axis=0)
            col[inside] = (0.2, 1.0, 0.2); n_in = int(inside.sum())
        pcd.colors = o3d.utility.Vector3dVector(col)
        vis.clear_geometries(); vis.add_geometry(pcd, reset_bounding_box=st['first'])
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0), reset_bounding_box=False)
        if keep.any():
            vis.add_geometry(boxes_mesh(b[keep], (0.1, 1.0, 0.1), args.line_radius), reset_bounding_box=False)
        if st['first']:
            if os.path.exists(args.view_file):
                vis.get_view_control().convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters(args.view_file), allow_arbitrary=True)
            else:
                top_view(vis)
        st['first'] = False
        print('[%d/%d] %s | %d points | %d detections >= %.2f (%d points inside boxes; scores: %s)' % (
            i + 1, len(ds), os.path.basename(ds.sample_file_list[i]), len(pts), keep.sum(), st['score'], n_in,
            ' '.join('%.2f' % x for x in sorted(s[keep], reverse=True)[:12])), flush=True)

    def top_view(v):
        """bird's-eye view, sensor-centred, zoomed so that a 200 m scene fills the window (key T)"""
        vc = v.get_view_control(); vc.set_front([0.0, 0.0, 1.0]); vc.set_up([0.0, 1.0, 0.0]); vc.set_lookat([0.0, 0.0, 0.0]); vc.set_zoom(0.45)
        return False

    def step(k):
        def cb(_):
            st['i'] = (st['i'] + k) % len(ds); show(); st['last'] = time.time(); return True
        return cb

    def thr(d):
        def cb(_):
            st['score'] = float(np.clip(st['score'] + d, 0.05, 0.95)); show(); return True
        return cb

    def toggle(_):
        st['play'] = not st['play']; print('autoplay' if st['play'] else 'paused'); return False

    def save_view(v):
        o3d.io.write_pinhole_camera_parameters(args.view_file, v.get_view_control().convert_to_pinhole_camera_parameters()); print('camera saved'); return False

    def load_view(v):
        if os.path.exists(args.view_file):
            v.get_view_control().convert_from_pinhole_camera_parameters(o3d.io.read_pinhole_camera_parameters(args.view_file), allow_arbitrary=True)
        return False

    def tick(v):
        if st['play'] and time.time() - st['last'] >= args.interval:
            st['i'] = (st['i'] + 1) % len(ds); show(); st['last'] = time.time(); return True
        return False

    for key, cb in ((ord('N'), step(1)), (262, step(1)), (ord('B'), step(-1)), (263, step(-1)), (ord('='), thr(0.05)), (ord('+'), thr(0.05)),
                    (ord('-'), thr(-0.05)), (ord(' '), toggle), (ord('V'), save_view), (ord('L'), load_view), (ord('T'), top_view)):
        vis.register_key_callback(key, cb)
    vis.register_animation_callback(tick)
    show(); st['last'] = time.time(); vis.run(); vis.destroy_window()


if __name__ == '__main__':
    main()
