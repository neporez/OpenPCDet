"""
Rotation helpers for 9-DoF boxes [x, y, z, dx, dy, dz, yaw, roll, pitch] (radians).

Two Euler conventions appear in the KRISO sim data:

  sim (as written by the Isaac Sim labeler, see view_buoyancy_labels.py):
      p_world = R @ p_local,  R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
  std (used inside OpenPCDet for 9-DoF; aerospace yaw-pitch-roll):
      p_world = R @ p_local,  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

With the std convention a world-z rotation only adds to yaw, and a y-flip is
(yaw, roll, pitch) -> (-yaw, -roll, pitch). Labels are converted sim -> std once, when the
dataset is loaded; every other function in this file works on the std convention unless noted.

Column layout stays [..., yaw, roll, pitch] in both conventions; only the meaning of the angles
changes.
"""
import numpy as np


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([o, z, z, z, c, -s, z, s, c], axis=-1).reshape(a.shape + (3, 3))


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([c, z, s, z, o, z, -s, z, c], axis=-1).reshape(a.shape + (3, 3))


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    o, z = np.ones_like(a), np.zeros_like(a)
    return np.stack([c, -s, z, s, c, z, z, z, o], axis=-1).reshape(a.shape + (3, 3))


def sim_euler_to_matrix(yaw, roll, pitch):
    """sim convention: R = Rx(roll) @ Ry(pitch) @ Rz(yaw). Inputs: arrays of the same shape."""
    yaw, roll, pitch = (np.asarray(v, dtype=np.float64) for v in (yaw, roll, pitch))
    return _rx(roll) @ _ry(pitch) @ _rz(yaw)


def std_euler_to_matrix(yaw, roll, pitch):
    """std convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    yaw, roll, pitch = (np.asarray(v, dtype=np.float64) for v in (yaw, roll, pitch))
    return _rz(yaw) @ _ry(pitch) @ _rx(roll)


def matrix_to_std_euler(rot):
    """Inverse of std_euler_to_matrix. rot: (..., 3, 3) -> yaw, roll, pitch (each (...,)).

    pitch is returned in [-pi/2, pi/2]; at the gimbal lock (|pitch| = pi/2) roll is set to 0.
    """
    rot = np.asarray(rot, dtype=np.float64)
    pitch = -np.arcsin(np.clip(rot[..., 2, 0], -1.0, 1.0))
    cos_p = np.cos(pitch)
    locked = cos_p < 1e-6
    safe = np.where(locked, 1.0, cos_p)
    yaw = np.where(locked, np.arctan2(-rot[..., 0, 1], rot[..., 1, 1]), np.arctan2(rot[..., 1, 0] / safe, rot[..., 0, 0] / safe))
    roll = np.where(locked, 0.0, np.arctan2(rot[..., 2, 1] / safe, rot[..., 2, 2] / safe))
    return yaw, roll, pitch


def sim_boxes_to_std(boxes):
    """(N, >=9) boxes with sim-convention angles -> copy with std-convention angles."""
    boxes = np.array(boxes, copy=True)
    if boxes.shape[0] == 0:
        return boxes
    rot = sim_euler_to_matrix(boxes[:, 6], boxes[:, 7], boxes[:, 8])
    boxes[:, 6], boxes[:, 7], boxes[:, 8] = matrix_to_std_euler(rot)
    return boxes


def boxes_to_rotation_matrices(boxes):
    """(N, >=7) std-convention boxes -> (N, 3, 3). Boxes with only 7 columns get roll = pitch = 0."""
    boxes = np.asarray(boxes)
    zeros = np.zeros(boxes.shape[0])
    roll = boxes[:, 7] if boxes.shape[1] > 7 else zeros
    pitch = boxes[:, 8] if boxes.shape[1] > 8 else zeros
    return std_euler_to_matrix(boxes[:, 6], roll, pitch)


def boxes_to_corners_9dof(boxes):
    """(N, >=7) std-convention boxes -> (N, 8, 3) corners, same corner order as
    box_utils.boxes_to_corners_3d (so the two agree when roll = pitch = 0)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    template = np.array([
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    ]) / 2
    local = boxes[:, None, 3:6] * template[None, :, :]
    rot = boxes_to_rotation_matrices(boxes)
    return np.einsum('nij,nkj->nki', rot, local) + boxes[:, None, 0:3]


def points_in_boxes_9dof(points, boxes):
    """points: (M, >=3), boxes: (N, >=7) std convention -> bool mask (N, M).

    Rotate the points into each box frame (R^T (p - c)) and compare with the half extents.
    """
    points = np.asarray(points, dtype=np.float64)[:, :3]
    boxes = np.asarray(boxes, dtype=np.float64)
    rot = boxes_to_rotation_matrices(boxes)
    shifted = points[None, :, :] - boxes[:, None, 0:3]
    local = np.einsum('nji,nmj->nmi', rot, shifted)
    return np.all(np.abs(local) <= boxes[:, None, 3:6] / 2, axis=-1)


def count_points_in_boxes_9dof(points, boxes, chunk=64):
    """(N,) number of points inside each box; chunked over boxes to bound memory."""
    boxes = np.asarray(boxes)
    counts = np.zeros(boxes.shape[0], dtype=np.int64)
    for start in range(0, boxes.shape[0], chunk):
        counts[start:start + chunk] = points_in_boxes_9dof(points, boxes[start:start + chunk]).sum(axis=1)
    return counts
