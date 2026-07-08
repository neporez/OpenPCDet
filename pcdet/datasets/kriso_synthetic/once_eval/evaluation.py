"""
Evaluation Server
Written by Jiageng Mao
Modified: Added BEV evaluation alongside 3D (tables split top/bottom)
"""

import numpy as np
import numba

from .iou_utils import rotate_iou_gpu_eval
from .eval_utils import compute_split_parts, overall_filter, distance_filter, overall_distance_filter

iou_threshold_dict = {
    'boat': 0.5,
}

superclass_iou_threshold_dict = {
    'boat': 0.5,
}


# ───────────────────────────────────────────────
#  IoU kernels
# ───────────────────────────────────────────────

def iou3d_kernel(gt_boxes, pred_boxes):
    """
    3D IoU (with height intersection, no heading filter)

    Args:
        gt_boxes:   [N, 7] (x, y, z, w, l, h, rot)
        pred_boxes: [M, 7]
    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(
        gt_boxes[:, [0, 1, 3, 4, 6]],
        pred_boxes[:, [0, 1, 3, 4, 6]],
        criterion=2
    )
    gt_max_h   = gt_boxes[:, [2]]   + gt_boxes[:, [5]]   * 0.5
    gt_min_h   = gt_boxes[:, [2]]   - gt_boxes[:, [5]]   * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol   = gt_boxes[:, [3]]   * gt_boxes[:, [4]]   * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    iou3d    = intersection_3d / union_3d
    return iou3d


def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """3D IoU with heading filter (diff_rot > 90° → IoU=0)"""
    iou3d = iou3d_kernel(gt_boxes, pred_boxes)
    diff_rot         = np.abs(gt_boxes[:, [6]] - pred_boxes[:, [6]].T)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi]
    iou3d[diff_rot > np.pi / 2] = 0
    return iou3d


def iou_bev_kernel(gt_boxes, pred_boxes):
    """
    BEV IoU (height axis ignored)

    Args:
        gt_boxes:   [N, 7]
        pred_boxes: [M, 7]
    Returns:
        iou_bev: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(
        gt_boxes[:, [0, 1, 3, 4, 6]],
        pred_boxes[:, [0, 1, 3, 4, 6]],
        criterion=2
    )
    gt_area   = gt_boxes[:, [3]]   * gt_boxes[:, [4]]
    pred_area = pred_boxes[:, [3]] * pred_boxes[:, [4]]
    union_2d  = gt_area + pred_area.T - intersection_2d
    iou_bev   = intersection_2d / union_2d
    return iou_bev


def iou_bev_kernel_with_heading(gt_boxes, pred_boxes):
    """BEV IoU with heading filter"""
    iou_bev          = iou_bev_kernel(gt_boxes, pred_boxes)
    diff_rot         = np.abs(gt_boxes[:, [6]] - pred_boxes[:, [6]].T)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi]
    iou_bev[diff_rot > np.pi / 2] = 0
    return iou_bev


# ───────────────────────────────────────────────
#  IoU batch computation
# ───────────────────────────────────────────────

def compute_ious(gt_annos, pred_annos, split_parts, with_heading, mode='3d'):
    """
    Args:
        mode: '3d' | 'bev'
    Returns:
        ious: list of [N_gt, N_pred] arrays per sample
    """
    assert mode in ('3d', 'bev')
    gt_num_per_sample   = np.stack([len(a["name"]) for a in gt_annos], 0)
    pred_num_per_sample = np.stack([len(a["name"]) for a in pred_annos], 0)

    if mode == '3d':
        kernel = iou3d_kernel_with_heading if with_heading else iou3d_kernel
    else:
        kernel = iou_bev_kernel_with_heading if with_heading else iou_bev_kernel

    ious       = []
    sample_idx = 0
    for num_part_samples in split_parts:
        gt_part   = gt_annos[sample_idx:sample_idx + num_part_samples]
        pred_part = pred_annos[sample_idx:sample_idx + num_part_samples]
        gt_boxes   = np.concatenate([a["boxes_3d"] for a in gt_part],   0)
        pred_boxes = np.concatenate([a["boxes_3d"] for a in pred_part], 0)

        iou_part = kernel(gt_boxes, pred_boxes)

        gt_idx, pred_idx = 0, 0
        for idx in range(num_part_samples):
            gn = gt_num_per_sample[sample_idx + idx]
            pn = pred_num_per_sample[sample_idx + idx]
            ious.append(iou_part[gt_idx:gt_idx + gn, pred_idx:pred_idx + pn])
            gt_idx   += gn
            pred_idx += pn
        sample_idx += num_part_samples
    return ious


# ───────────────────────────────────────────────
#  Main evaluation entry point
# ───────────────────────────────────────────────

def get_evaluation_results(gt_annos, pred_annos, classes,
                           use_superclass=True,
                           iou_thresholds=None,
                           num_pr_points=50,
                           difficulty_mode='Overall&Distance',
                           ap_with_heading=True,
                           num_parts=100,
                           print_ok=False):
    """
    Evaluate both 3D AP and BEV AP simultaneously.
    Output tables are printed one above the other.
    """
    if iou_thresholds is None:
        iou_thresholds = superclass_iou_threshold_dict if use_superclass else iou_threshold_dict

    assert len(gt_annos) == len(pred_annos), "GT / pred sample count mismatch"
    assert difficulty_mode in ['Overall&Distance', 'Overall', 'Distance']

    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)

    # ── compute both IoU sets ──────────────────
    ious_3d  = compute_ious(gt_annos, pred_annos, split_parts, ap_with_heading, mode='3d')
    ious_bev = compute_ious(gt_annos, pred_annos, split_parts, ap_with_heading, mode='bev')
    mode_ious_list = [ious_3d, ious_bev]          # index 0 → 3D, 1 → BEV

    # ── difficulty setup ──────────────────────
    num_classes = len(classes)
    if difficulty_mode == 'Distance':
        num_difficulties = 3
        difficulty_types = ['0-100m', '100-150m', '150m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    else:  # Overall&Distance
        num_difficulties = 4
        difficulty_types = ['overall', '0-100m', '100-150m', '150m-inf']

    # ── precision / recall: shape [2, C, D, P+1] ──
    #    axis-0: 0=3D, 1=BEV
    precision = np.zeros([2, num_classes, num_difficulties, num_pr_points + 1])
    recall    = np.zeros([2, num_classes, num_difficulties, num_pr_points + 1])

    for mode_idx, ious in enumerate(mode_ious_list):
        for cls_idx, cur_class in enumerate(classes):
            iou_threshold = iou_thresholds[cur_class]
            for diff_idx in range(num_difficulties):

                # ── collect scores & flags ────────
                accum_all_scores, gt_flags, pred_flags = [], [], []
                num_valid_gt = 0
                for sample_idx in range(num_samples):
                    gt_anno   = gt_annos[sample_idx]
                    pred_anno = pred_annos[sample_idx]
                    pred_score = pred_anno['score']
                    iou        = ious[sample_idx]

                    gt_flag, pred_flag = filter_data(
                        gt_anno, pred_anno, difficulty_mode,
                        difficulty_level=diff_idx,
                        class_name=cur_class,
                        use_superclass=use_superclass
                    )
                    gt_flags.append(gt_flag)
                    pred_flags.append(pred_flag)
                    num_valid_gt += sum(gt_flag == 0)

                    accum_scores = accumulate_scores(
                        iou, pred_score, gt_flag, pred_flag,
                        iou_threshold=iou_threshold
                    )
                    accum_all_scores.append(accum_scores)

                all_scores = np.concatenate(accum_all_scores, axis=0)
                thresholds = get_thresholds(all_scores, num_valid_gt,
                                            num_pr_points=num_pr_points)

                # ── confusion matrix ──────────────
                confusion_matrix = np.zeros([len(thresholds), 3])
                for sample_idx in range(num_samples):
                    pred_score = pred_annos[sample_idx]['score']
                    iou        = ious[sample_idx]
                    gt_flag    = gt_flags[sample_idx]
                    pred_flag  = pred_flags[sample_idx]
                    for th_idx, score_th in enumerate(thresholds):
                        tp, fp, fn = compute_statistics(
                            iou, pred_score, gt_flag, pred_flag,
                            score_threshold=score_th,
                            iou_threshold=iou_threshold
                        )
                        confusion_matrix[th_idx, 0] += tp
                        confusion_matrix[th_idx, 1] += fp
                        confusion_matrix[th_idx, 2] += fn

                # ── P-R curve ─────────────────────
                for th_idx in range(len(thresholds)):
                    tp_ = confusion_matrix[th_idx, 0]
                    fp_ = confusion_matrix[th_idx, 1]
                    fn_ = confusion_matrix[th_idx, 2]
                    recall   [mode_idx, cls_idx, diff_idx, th_idx] = tp_ / (tp_ + fn_)
                    precision[mode_idx, cls_idx, diff_idx, th_idx] = tp_ / (tp_ + fp_)

                # monotone envelope
                for th_idx in range(len(thresholds)):
                    precision[mode_idx, cls_idx, diff_idx, th_idx] = np.max(
                        precision[mode_idx, cls_idx, diff_idx, th_idx:])
                    recall[mode_idx, cls_idx, diff_idx, th_idx] = np.max(
                        recall[mode_idx, cls_idx, diff_idx, th_idx:])

    # ── AP: shape [2, num_classes, num_difficulties] ──
    AP = np.sum(precision[..., 1:], axis=-1) / num_pr_points * 100

    # ── build return dict ─────────────────────
    mode_labels = ['3D', 'BEV']
    ret_dict    = {}
    for mode_idx, mlabel in enumerate(mode_labels):
        for cls_idx, cur_class in enumerate(classes):
            for diff_idx, diff_type in enumerate(difficulty_types):
                key = f'AP_{mlabel}_{cur_class}/{diff_type}'
                ret_dict[key] = AP[mode_idx, cls_idx, diff_idx]
    mAP = np.mean(AP, axis=1)   # [2, num_difficulties]
    for mode_idx, mlabel in enumerate(mode_labels):
        for diff_idx, diff_type in enumerate(difficulty_types):
            key = f'AP_{mlabel}_mean/{diff_type}'
            ret_dict[key] = mAP[mode_idx, diff_idx]

    # ── build result string (3D table / BEV table) ──
    ret_str = _build_table_str(AP, mAP, classes, difficulty_types,
                               num_difficulties, num_pr_points, mode_labels)

    if print_ok:
        print(ret_str)

    return ret_str, ret_dict


def _build_table_str(AP, mAP, classes, difficulty_types,
                     num_difficulties, num_pr_points, mode_labels):
    """
    3D 테이블과 BEV 테이블을 위아래로 출력
    """
    col_w   = 12   # column width
    sep     = '+' + '-' * (col_w + 2)

    def make_table(mode_idx, mlabel):
        header_sep = sep + (sep * num_difficulties) + '+'
        lines      = []
        lines.append(f'\n[ {mlabel} AP@{num_pr_points} ]')
        lines.append(header_sep)

        # header row
        hdr = '| {:<{w}} |'.format('Class', w=col_w)
        for dt in difficulty_types:
            hdr += ' {:<{w}} |'.format(dt, w=col_w)
        lines.append(hdr)
        lines.append(header_sep)

        # per-class rows
        for cls_idx, cur_class in enumerate(classes):
            row = '| {:<{w}} |'.format(cur_class, w=col_w)
            for diff_idx in range(num_difficulties):
                row += ' {:<{w}.2f} |'.format(AP[mode_idx, cls_idx, diff_idx], w=col_w)
            lines.append(row)

        lines.append(header_sep)

        # mAP row
        row = '| {:<{w}} |'.format('mAP', w=col_w)
        for diff_idx in range(num_difficulties):
            row += ' {:<{w}.2f} |'.format(mAP[mode_idx, diff_idx], w=col_w)
        lines.append(row)
        lines.append(header_sep)
        return '\n'.join(lines)

    parts = [make_table(i, lbl) for i, lbl in enumerate(mode_labels)]
    return '\n'.join(parts) + '\n'


# ───────────────────────────────────────────────
#  Threshold / score helpers  (unchanged)
# ───────────────────────────────────────────────

@numba.jit(nopython=True)
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores.sort()
    scores = scores[::-1]
    recall_level = 0
    thresholds   = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points
        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds


@numba.jit(nopython=True)
def accumulate_scores(iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt    = iou.shape[0]
    num_pred  = iou.shape[1]
    assigned  = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_idx    = 0
    for i in range(num_gt):
        if gt_flag[i] == -1:
            continue
        det_idx        = -1
        detected_score = -1
        for j in range(num_pred):
            if pred_flag[j] == -1:
                continue
            if assigned[j]:
                continue
            iou_ij     = iou[i, j]
            pred_score = pred_scores[j]
            if (iou_ij > iou_threshold) and (pred_score > detected_score):
                det_idx        = j
                detected_score = pred_score
        if (detected_score == -1) and (gt_flag[i] == 0):
            pass
        elif (detected_score != -1) and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):
            assigned[det_idx] = True
        elif detected_score != -1:
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_idx += 1
            assigned[det_idx] = True
    return accum_scores[:accum_idx]


@numba.jit(nopython=True)
def compute_statistics(iou, pred_scores, gt_flag, pred_flag,
                       score_threshold, iou_threshold):
    num_gt   = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold

    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1:
            continue
        det_idx              = -1
        detected             = False
        best_matched_iou     = 0
        gt_assigned_to_ignore = False

        for j in range(num_pred):
            if pred_flag[j] == -1:
                continue
            if assigned[j]:
                continue
            if under_threshold[j]:
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and \
               (iou_ij > best_matched_iou or gt_assigned_to_ignore) and \
               pred_flag[j] == 0:
                best_matched_iou      = iou_ij
                det_idx               = j
                detected              = True
                gt_assigned_to_ignore = False
            elif (iou_ij > iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx               = j
                detected              = True
                gt_assigned_to_ignore = True

        if (not detected) and gt_flag[i] == 0:
            fn += 1
        elif detected and (gt_flag[i] == 1 or pred_flag[det_idx] == 1):
            assigned[det_idx] = True
        elif detected:
            tp += 1
            assigned[det_idx] = True

    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or
                pred_flag[j] == 1 or under_threshold[j]):
            fp += 1

    return tp, fp, fn


def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level,
                class_name, use_superclass):
    """
    Returns gt_flag, pred_flag
        0  : accepted
        1  : same class but ignored (difficulty)
       -1  : rejected (different class)
    """
    num_gt  = len(gt_anno['name'])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_or(gt_anno['name'] == 'Pedestrian',
                                   gt_anno['name'] == 'Cyclist')
        else:
            reject = gt_anno['name'] != class_name
    else:
        reject = gt_anno['name'] != class_name
    gt_flag[reject] = -1

    num_pred  = len(pred_anno['name'])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_or(pred_anno['name'] == 'Pedestrian',
                                   pred_anno['name'] == 'Cyclist')
        else:
            reject = pred_anno['name'] != class_name
    else:
        reject = pred_anno['name'] != class_name
    pred_flag[reject] = -1

    if difficulty_mode == 'Overall':
        gt_flag[overall_filter(gt_anno['boxes_3d'])]     = 1
        pred_flag[overall_filter(pred_anno['boxes_3d'])] = 1
    elif difficulty_mode == 'Distance':
        gt_flag[distance_filter(gt_anno['boxes_3d'],   difficulty_level)] = 1
        pred_flag[distance_filter(pred_anno['boxes_3d'], difficulty_level)] = 1
    elif difficulty_mode == 'Overall&Distance':
        gt_flag[overall_distance_filter(gt_anno['boxes_3d'],   difficulty_level)] = 1
        pred_flag[overall_distance_filter(pred_anno['boxes_3d'], difficulty_level)] = 1
    else:
        raise NotImplementedError

    return gt_flag, pred_flag