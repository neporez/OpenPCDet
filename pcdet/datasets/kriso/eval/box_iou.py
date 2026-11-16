"""
Exact 3D IoU of oriented boxes with roll / pitch (9-DoF boxes, std Euler convention of
utils/box_rotation_utils.py).

The intersection of two boxes is a convex polyhedron whose vertices are
  * corners of A inside B, corners of B inside A, and
  * intersections of the edges of A with the faces of B (and the other way round);
the volume is that of the convex hull of these points. Pairs whose circumscribed spheres do not touch,
or whose headings differ by more than 90 degrees (when requested), are skipped (IoU = 0).
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ....utils import box_rotation_utils

_EDGES = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
_EPS = 1e-9


class _Box:
    """Rotation, center, half extents and corners of one box, computed once."""

    def __init__(self, box):
        box = np.asarray(box, dtype=np.float64)
        self.rot = box_rotation_utils.boxes_to_rotation_matrices(box[None])[0]
        self.center = box[:3]
        self.half = box[3:6] / 2
        self.corners = box_rotation_utils.boxes_to_corners_9dof(box[None])[0]
        self.volume = float(np.prod(box[3:6]))

    def contains(self, points):
        local = (points - self.center) @ self.rot
        return np.all(np.abs(local) <= self.half + _EPS, axis=1)

    def edge_face_points(self, other):
        """Points where the edges of `other` cross the faces of this box (inside the face)."""
        p0, p1 = other.corners[_EDGES[:, 0]], other.corners[_EDGES[:, 1]]
        l0, l1 = (p0 - self.center) @ self.rot, (p1 - self.center) @ self.rot
        delta = l1 - l0
        found = []
        for k in range(3):
            others = [i for i in range(3) if i != k]
            for side in (-1.0, 1.0):
                den = delta[:, k]
                ok = np.abs(den) > 1e-12
                t = np.where(ok, (side * self.half[k] - l0[:, k]) / np.where(ok, den, 1.0), -1.0)
                valid = ok & (t >= 0) & (t <= 1)
                if not valid.any():
                    continue
                local = l0[valid] + t[valid, None] * delta[valid]
                good = np.all(np.abs(local[:, others]) <= self.half[others] + _EPS, axis=1)
                if good.any():
                    found.append(self.center + local[good] @ self.rot.T)
        return found


def intersection_volume(a, b):
    points = [a.corners[b.contains(a.corners)], b.corners[a.contains(b.corners)]]
    points += b.edge_face_points(a)
    points += a.edge_face_points(b)
    points = np.concatenate([p for p in points if len(p)], axis=0) if any(len(p) for p in points) else np.zeros((0, 3))
    if len(points) < 4:
        return 0.0
    try:
        return float(ConvexHull(points).volume)
    except QhullError:  # flat / degenerate intersection
        return 0.0


def iou3d_matrix(gt_boxes, pred_boxes, with_heading=True):
    """(N, 9+) x (M, 9+) -> (N, M) IoU. with_heading: pairs with a heading difference > 90 deg get IoU 0."""
    gt_boxes = np.asarray(gt_boxes, dtype=np.float64)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    iou = np.zeros((len(gt_boxes), len(pred_boxes)))
    if iou.size == 0:
        return iou

    gt_radius = np.linalg.norm(gt_boxes[:, 3:6], axis=1) / 2
    pred_radius = np.linalg.norm(pred_boxes[:, 3:6], axis=1) / 2
    dist = np.linalg.norm(gt_boxes[:, None, :3] - pred_boxes[None, :, :3], axis=-1)
    candidate = dist <= gt_radius[:, None] + pred_radius[None, :]
    if with_heading:
        diff = np.abs(gt_boxes[:, None, 6] - pred_boxes[None, :, 6])
        diff = np.where(diff >= np.pi, 2 * np.pi - diff, diff)
        candidate &= diff <= np.pi / 2

    gt_cache, pred_cache = {}, {}
    for i, j in zip(*np.nonzero(candidate)):
        a = gt_cache.setdefault(i, _Box(gt_boxes[i]))
        b = pred_cache.setdefault(j, _Box(pred_boxes[j]))
        inter = intersection_volume(a, b)
        iou[i, j] = inter / (a.volume + b.volume - inter)
    return iou
