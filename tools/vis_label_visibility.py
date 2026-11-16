"""
Play the real KRISO frames in Open3D and colour every label by the visibility rule.

    visibility ratio = observed points / expected points,   expected = c * visible_area / distance^2
    visible_area     = length*height*|sin(aspect)| + width*height*|cos(aspect)|   (aspect = line of sight - heading)
    c                = points per steradian from LABEL_FILTER.SENSOR (or a fixed LABEL_FILTER.C)

colours:  green = used as a label,  red = not a label
rule:     LABEL_FILTER of the dataset config (pcdet/utils/visibility_utils.py): a box is a label if it has at least
          ABS_POINTS points or its visibility ratio is >= RATIO; --ratio / --abs_points override them
points are white. The terminal prints, per frame, every red label (length, distance, points, expected, ratio).

keys:  space = pause / resume,  N or Right = next frame,  B or Left = previous frame,  Q / Esc = quit
       V = save the camera to --view_file,  L = load it

python vis_label_visibility.py                       # frames that contain at least one red label
python vis_label_visibility.py --all_frames --interval 0.2 --sequence ulsan_1105
"""
import _init_path
import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import open3d

import yaml

from pcdet.utils import box_rotation_utils, visibility_utils

ROOT = Path(__file__).resolve().parent.parent
EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
COLORS = {'label': (0.1, 1.0, 0.1), 'not_label': (1.0, 0.15, 0.15)}


def classify(info, cfg):
    boxes = info['annos']['boxes_3d'].astype(np.float64)
    pts = info['annos']['num_points_in_gt'].astype(np.float64)
    dist = np.linalg.norm(boxes[:, :3], axis=1)
    expected = visibility_utils.expected_points(boxes, visibility_utils.get_c(cfg))
    ratio = visibility_utils.visibility_ratio(boxes, pts, visibility_utils.get_c(cfg))
    status = np.where(visibility_utils.is_label(boxes, pts, cfg), 'label', 'not_label')
    return boxes, pts, dist, expected, ratio, status


def box_lineset(boxes, status):
    points, lines, colors = [], [], []
    corners = box_rotation_utils.boxes_to_corners_9dof(boxes) if len(boxes) else np.zeros((0, 8, 3))
    rot = box_rotation_utils.boxes_to_rotation_matrices(boxes) if len(boxes) else np.zeros((0, 3, 3))
    for b, cor, R, s in zip(boxes, corners, rot, status):
        base = len(points)
        points.extend(cor.tolist())
        lines.extend([[base + i, base + j] for i, j in EDGES])
        colors.extend([COLORS[s]] * len(EDGES))
        front = b[:3] + R @ np.array([b[3] / 2, 0.0, 0.0])  # heading line from the centre to the front face
        points.extend([b[:3].tolist(), front.tolist()])
        lines.append([base + 8, base + 9])
        colors.append(COLORS[s])
    ls = open3d.geometry.LineSet()
    if points:
        ls.points = open3d.utility.Vector3dVector(np.array(points))
        ls.lines = open3d.utility.Vector2iVector(np.array(lines))
        ls.colors = open3d.utility.Vector3dVector(np.array(colors))
    return ls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--infos', default=str(ROOT / 'data/kriso_ulsan/kriso_ulsan_infos_real.pkl'))
    parser.add_argument('--cfg_file', default=str(ROOT / 'tools/cfgs/dataset_configs/kriso_dataset.yaml'), help='LABEL_FILTER is read from here')
    parser.add_argument('--ratio', type=float, default=None, help='override LABEL_FILTER.RATIO')
    parser.add_argument('--abs_points', type=int, default=None, help='override LABEL_FILTER.ABS_POINTS (-1 = ratio only)')
    parser.add_argument('--interval', type=float, default=0.2, help='seconds per frame')
    parser.add_argument('--all_frames', action='store_true', help='play every frame, not only the ones with a red label')
    parser.add_argument('--sequence', default=None, help='only this sequence (ulsan_0609 or ulsan_1105)')
    parser.add_argument('--start', type=int, default=0, help='index of the first frame to show (after filtering)')
    parser.add_argument('--point_size', type=float, default=2.0)
    parser.add_argument('--view_file', default=str(ROOT / 'output' / 'vis_view.json'))
    parser.add_argument('--max_frames', type=int, default=None, help='stop after this many frames (testing)')
    args = parser.parse_args()

    infos = pickle.load(open(args.infos, 'rb'))
    cfg = dict(yaml.safe_load(open(args.cfg_file))['LABEL_FILTER'])
    if args.ratio is not None:
        cfg['RATIO'] = args.ratio
    if args.abs_points is not None:
        cfg['ABS_POINTS'] = None if args.abs_points < 0 else args.abs_points
    c = visibility_utils.get_c(cfg)
    if args.sequence:
        infos = [i for i in infos if i['sequence_id'] == args.sequence]
    frames = []
    totals = {k: 0 for k in COLORS}
    for info in infos:
        res = classify(info, cfg)
        for k in COLORS:
            totals[k] += int(np.sum(res[5] == k))
        if args.all_frames or np.any(res[5] == 'not_label'):
            frames.append((info, res))
    rule = 'ratio < %.2f' % cfg['RATIO'] + ('' if cfg.get('ABS_POINTS') is None else ' and points < %d' % cfg['ABS_POINTS'])
    print('c = %.0f | labels: green %d, red (%s) %d | %d frames to show' % (c, totals['label'], rule, totals['not_label'], len(frames)))
    if not frames:
        return
    data_root = Path(args.infos).parent / 'data'

    state = {'idx': max(0, min(args.start, len(frames) - 1)), 'paused': False, 'last': 0.0, 'shown': 0, 'first': True}
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name='label visibility (green = label, red = not a label: %s)' % rule)
    opt = vis.get_render_option()
    opt.point_size = args.point_size
    opt.background_color = np.array([0.05, 0.05, 0.05])

    def show(idx):
        info, (boxes, pts, dist, expected, ratio, status) = frames[idx]
        raw = np.fromfile(data_root / info['sequence_id'] / 'lidar_roof' / ('%s.bin' % info['frame_id']), dtype=np.float32).reshape(-1, 4)[:, :3]
        raw = raw[np.isfinite(raw).all(axis=1)]
        pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(raw.astype(np.float64)))
        pcd.paint_uniform_color([0.85, 0.85, 0.85])
        vis.clear_geometries()
        vis.add_geometry(pcd, reset_bounding_box=state['first'])
        vis.add_geometry(box_lineset(boxes, status), reset_bounding_box=False)
        vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0), reset_bounding_box=False)
        if state['first'] and os.path.exists(args.view_file):
            vis.get_view_control().convert_from_pinhole_camera_parameters(open3d.io.read_pinhole_camera_parameters(args.view_file), allow_arbitrary=True)
        state['first'] = False
        line = '[%d/%d] %s/%s: %d boxes, green %d, red %d' % (
            idx + 1, len(frames), info['sequence_id'], info['frame_id'], len(boxes),
            np.sum(status == 'label'), np.sum(status == 'not_label'))
        print(line)
        for k in np.where(status == 'not_label')[0]:
            print('      red  length %5.1f m  dist %5.1f m  points %5d  expected %7.1f  ratio %.3f'
                  % (boxes[k, 3], dist[k], pts[k], expected[k], ratio[k]))
        state['shown'] += 1

    def step(delta):
        def cb(v):
            state['idx'] = (state['idx'] + delta) % len(frames)
            show(state['idx'])
            state['last'] = time.time()
            return True
        return cb

    def toggle(v):
        state['paused'] = not state['paused']
        print('paused' if state['paused'] else 'playing')
        return False

    def save_view(v):
        open3d.io.write_pinhole_camera_parameters(args.view_file, v.get_view_control().convert_to_pinhole_camera_parameters())
        print('camera saved to', args.view_file)
        return False

    def load_view(v):
        if os.path.exists(args.view_file):
            v.get_view_control().convert_from_pinhole_camera_parameters(open3d.io.read_pinhole_camera_parameters(args.view_file), allow_arbitrary=True)
        return False

    def tick(v):
        if args.max_frames is not None and state['shown'] >= args.max_frames:
            v.close()
            return False
        if not state['paused'] and time.time() - state['last'] >= args.interval:
            state['idx'] = (state['idx'] + 1) % len(frames)
            show(state['idx'])
            state['last'] = time.time()
            return True
        return False

    for key, cb in ((ord(' '), toggle), (ord('N'), step(1)), (262, step(1)), (ord('B'), step(-1)), (263, step(-1)),
                    (ord('V'), save_view), (ord('L'), load_view)):
        vis.register_key_callback(key, cb)
    vis.register_animation_callback(tick)
    show(state['idx'])
    state['last'] = time.time()
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    main()
