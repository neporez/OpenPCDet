"""
Open3d visualization tool box
Written by Jihan YANG
All rights preserved from 2021 - present.
"""
import os
from functools import partial

import open3d
import torch
import matplotlib
import numpy as np

from pcdet.utils.box_rotation_utils import std_euler_to_matrix

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]


def get_coor_colors(obj_labels):
    """
    Args:
        obj_labels: 1 is ground, labels > 1 indicates different instance cluster

    Returns:
        rgb: [N, 3]. color for each point.
    """
    colors = matplotlib.colors.XKCD_COLORS.values()
    max_color_num = obj_labels.max()

    color_list = list(colors)[:max_color_num+1]
    colors_rgba = [matplotlib.colors.to_rgba_array(color) for color in color_list]
    label_rgba = np.array(colors_rgba)[obj_labels]
    label_rgba = label_rgba.squeeze()[:, :3]

    return label_rgba


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None, draw_origin=True,
                point_size=1.0, line_radius=None, gt_color=(0, 0, 1), view_file=None):
    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    if isinstance(gt_boxes, torch.Tensor):
        gt_boxes = gt_boxes.cpu().numpy()
    if isinstance(ref_boxes, torch.Tensor):
        ref_boxes = ref_boxes.cpu().numpy()

    saved_view = None
    if view_file is not None and os.path.exists(view_file):
        saved_view = open3d.io.read_pinhole_camera_parameters(view_file)

    if view_file is None:
        vis = open3d.visualization.Visualizer()
    else:
        vis = open3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(ord('V'), partial(save_view, view_file=view_file))
        vis.register_key_callback(ord('L'), partial(load_view, view_file=view_file))
        print(f'[view] V: save the current camera to {view_file}, L: load it again')
    if saved_view is not None:  # the window must have the saved size, otherwise the saved view cannot be applied
        vis.create_window(width=saved_view.intrinsic.width, height=saved_view.intrinsic.height)
    else:
        vis.create_window()

    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        pts.colors = open3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = open3d.utility.Vector3dVector(point_colors)

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, gt_color, line_radius=line_radius)

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores, line_radius=line_radius)

    if saved_view is not None:
        vis.get_view_control().convert_from_pinhole_camera_parameters(saved_view, allow_arbitrary=True)

    vis.run()
    vis.destroy_window()


def save_view(vis, view_file):
    params = vis.get_view_control().convert_to_pinhole_camera_parameters()
    open3d.io.write_pinhole_camera_parameters(view_file, params)
    print(f'[view] saved camera to {view_file}')
    return False


def load_view(vis, view_file):
    if not os.path.exists(view_file):
        print(f'[view] {view_file} does not exist yet, press V to save the current camera first')
        return False
    params = open3d.io.read_pinhole_camera_parameters(view_file)
    vis.get_view_control().convert_from_pinhole_camera_parameters(params, allow_arbitrary=True)
    print(f'[view] loaded camera from {view_file}')
    return False


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    if len(gt_boxes) >= 9:  # 9-DoF box [.., yaw, roll, pitch], std convention (see utils/box_rotation_utils.py)
        rot = std_euler_to_matrix(gt_boxes[6], gt_boxes[7], gt_boxes[8])
    else:
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = open3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = open3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = open3d.utility.Vector2iVector(lines)

    return line_set, box3d


def line_set_to_cylinders(line_set, color, radius):
    """Open3D LineSet width is fixed to 1px, so draw each edge as a thin cylinder to get thick lines."""
    pts = np.asarray(line_set.points)
    meshes = []
    for a, b in np.asarray(line_set.lines):
        vec = pts[b] - pts[a]
        length = np.linalg.norm(vec)
        if length < 1e-6:
            continue
        cyl = open3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=8)
        direction = vec / length
        axis = np.cross([0, 0, 1], direction)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            angle = np.arccos(np.clip(direction[2], -1.0, 1.0))
            rot = open3d.geometry.get_rotation_matrix_from_axis_angle(axis / axis_norm * angle)
            cyl.rotate(rot, center=(0, 0, 0))
        elif direction[2] < 0:
            cyl.rotate(open3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0]), center=(0, 0, 0))
        cyl.translate((pts[a] + pts[b]) / 2)
        cyl.paint_uniform_color(color)
        meshes.append(cyl)
    return meshes


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, line_radius=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is not None:
            color_i = box_colormap[ref_labels[i]]
        else:
            color_i = color

        if line_radius is None:
            line_set.paint_uniform_color(color_i)
            vis.add_geometry(line_set)
        else:
            for mesh in line_set_to_cylinders(line_set, color_i, line_radius):
                vis.add_geometry(mesh)

        # if score is not None:
        #     corners = box3d.get_box_points()
        #     vis.add_3d_label(corners[5], '%.2f' % score[i])
    return vis
