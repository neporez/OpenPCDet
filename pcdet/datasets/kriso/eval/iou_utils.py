"""
Rotate IoU computation - CPU numpy implementation
(Replaced numba CUDA kernels for CUDA 13 compatibility)
"""
import math
import numpy as np


def div_up(m, n):
    return m // n + (m % n > 0)


def _trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])) / 2.0


def _area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(_trangle_area(int_pts[:2],
                                      int_pts[2 * i + 2:2 * i + 4],
                                      int_pts[2 * i + 4:2 * i + 6]))
    return area_val


def _sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = np.zeros(2, dtype=np.float32)
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        vs = np.zeros(16, dtype=np.float32)
        for i in range(num_of_inter):
            v0 = int_pts[2 * i] - center[0]
            v1 = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v0 * v0 + v1 * v1)
            v0 /= d
            v1 /= d
            if v1 < 0:
                v0 = -2 - v0
            vs[i] = v0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1
                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


def _line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = np.array([pts1[2 * i],         pts1[2 * i + 1]],         dtype=np.float32)
    B = np.array([pts1[2 * ((i+1)%4)], pts1[2 * ((i+1)%4) + 1]], dtype=np.float32)
    C = np.array([pts2[2 * j],         pts2[2 * j + 1]],         dtype=np.float32)
    D = np.array([pts2[2 * ((j+1)%4)], pts2[2 * ((j+1)%4) + 1]], dtype=np.float32)

    BA0 = B[0] - A[0]; BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]; CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]; CA1 = C[1] - A[1]

    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]; DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


def _point_in_quadrilateral(pt_x, pt_y, corners):
    PA0 = corners[0] - pt_x; PA1 = corners[1] - pt_y
    PB0 = corners[2] - pt_x; PB1 = corners[3] - pt_y
    PC0 = corners[4] - pt_x; PC1 = corners[5] - pt_y
    PD0 = corners[6] - pt_x; PD1 = corners[7] - pt_y
    PAB = PA0 * PB1 - PB0 * PA1
    PBC = PB0 * PC1 - PC0 * PB1
    PCD = PC0 * PD1 - PD0 * PC1
    PDA = PD0 * PA1 - PA0 * PD1
    return (PAB >= 0 and PBC >= 0 and PCD >= 0 and PDA >= 0) or \
           (PAB <= 0 and PBC <= 0 and PCD <= 0 and PDA <= 0)


def _quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if _point_in_quadrilateral(pts1[2*i], pts1[2*i+1], pts2):
            int_pts[num_of_inter*2]   = pts1[2*i]
            int_pts[num_of_inter*2+1] = pts1[2*i+1]
            num_of_inter += 1
        if _point_in_quadrilateral(pts2[2*i], pts2[2*i+1], pts1):
            int_pts[num_of_inter*2]   = pts2[2*i]
            int_pts[num_of_inter*2+1] = pts2[2*i+1]
            num_of_inter += 1
    temp_pts = np.zeros(2, dtype=np.float32)
    for i in range(4):
        for j in range(4):
            has_pts = _line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter*2]   = temp_pts[0]
                int_pts[num_of_inter*2+1] = temp_pts[1]
                num_of_inter += 1
    return num_of_inter


def _rbbox_to_corners(corners, rbbox):
    angle   = rbbox[4]
    a_cos   = math.cos(angle)
    a_sin   = math.sin(angle)
    center_x = rbbox[0]; center_y = rbbox[1]
    x_d = rbbox[2];      y_d = rbbox[3]
    corners_x = np.array([-x_d/2, -x_d/2,  x_d/2, x_d/2], dtype=np.float32)
    corners_y = np.array([-y_d/2,  y_d/2,  y_d/2, -y_d/2], dtype=np.float32)
    for i in range(4):
        corners[2*i]   = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2*i+1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


def _inter(rbbox1, rbbox2):
    corners1 = np.zeros(8, dtype=np.float32)
    corners2 = np.zeros(8, dtype=np.float32)
    int_pts  = np.zeros(16, dtype=np.float32)
    _rbbox_to_corners(corners1, rbbox1)
    _rbbox_to_corners(corners2, rbbox2)
    num_intersection = _quadrilateral_intersection(corners1, corners2, int_pts)
    _sort_vertex_in_convex_polygon(int_pts, num_intersection)
    return _area(int_pts, num_intersection)


def _devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = _inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Rotate IoU - CPU numpy implementation (CUDA 13 compatible)"""
    boxes       = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    for i in range(N):
        for j in range(K):
            iou[i, j] = _devRotateIoUEval(boxes[i], query_boxes[j], criterion)
    return iou.astype(boxes.dtype)