"""
Evaluation Server
Written by Jiageng Mao
Modified: Added BEV evaluation alongside 3D (tables split top/bottom)
kriso: per-sample IoU, exact 6-DoF 3D IoU for 9-DoF boxes (box_iou.py)
"""

import numpy as np
import numba

from .iou_utils import rotate_iou_gpu_eval
from .box_iou import iou3d_matrix
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
    IoU matrices per sample (only the pairs inside one sample are computed).

    Args:
        mode: '3d' | 'bev'. '3d' is the exact 6-DoF IoU when both sides have 9 box values
              (x, y, z, dx, dy, dz, yaw, roll, pitch), otherwise the yaw-only 3D IoU.
              'bev' only uses (x, y, dx, dy, yaw).
        split_parts: kept for call compatibility, not used.
    Returns:
        ious: list of [N_gt, N_pred] arrays per sample
    """
    assert mode in ('3d', 'bev')
    if mode == '3d':
        kernel = iou3d_kernel_with_heading if with_heading else iou3d_kernel
    else:
        kernel = iou_bev_kernel_with_heading if with_heading else iou_bev_kernel

    ious = []
    for gt_anno, pred_anno in zip(gt_annos, pred_annos):
        gt_boxes = np.asarray(gt_anno["boxes_3d"])
        pred_boxes = np.asarray(pred_anno["boxes_3d"])
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            ious.append(np.zeros((len(gt_boxes), len(pred_boxes))))
        elif mode == '3d' and gt_boxes.shape[1] >= 9 and pred_boxes.shape[1] >= 9:
            ious.append(iou3d_matrix(gt_boxes[:, :9], pred_boxes[:, :9], with_heading=with_heading))
        else:
            ious.append(kernel(gt_boxes[:, :7], pred_boxes[:, :7]))
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
                           print_ok=False,
                           count_score_thresh=0.3):
    """
    Evaluate both 3D AP and BEV AP simultaneously.
    Output tables are printed one above the other, followed by TP / FP / FN (+ precision, recall) of every bin at one
    operating point: predictions with score >= count_score_thresh (same matching as the AP: a gt or prediction that is
    "don't care" in a bin counts neither as FN nor as FP).
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
        num_difficulties = 4
        difficulty_types = ['0-100m', '100-150m', '0-150m', '150m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    else:  # Overall&Distance
        # length bins (levels 5 / 6, eval_utils._length_flag) restored 2026-11-13: Jangsaengpo big ships are reported separately
        difficulty_types = ['overall', '0-100m', '100-150m', '0-150m', '150m-inf', 'len0-40m', 'len40m-inf']
        num_difficulties = len(difficulty_types)

    # ── precision / recall: shape [2, C, D, P+1] ──
    #    axis-0: 0=3D, 1=BEV
    precision = np.zeros([2, num_classes, num_difficulties, num_pr_points + 1])
    recall    = np.zeros([2, num_classes, num_difficulties, num_pr_points + 1])
    num_valid_gt_all = np.zeros([2, num_classes, num_difficulties], dtype=np.int64)
    counts = np.zeros([2, num_classes, num_difficulties, 3], dtype=np.int64)  # TP, FP, FN at count_score_thresh

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

                num_valid_gt_all[mode_idx, cls_idx, diff_idx] = num_valid_gt
                for sample_idx in range(num_samples):
                    counts[mode_idx, cls_idx, diff_idx] += compute_statistics(
                        ious[sample_idx], pred_annos[sample_idx]['score'], gt_flags[sample_idx], pred_flags[sample_idx],
                        score_threshold=count_score_thresh, iou_threshold=iou_threshold)
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
    AP[num_valid_gt_all == 0] = np.nan  # no gt in this bin (e.g. no boat > 40 m): undefined, not 0

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

    for mode_idx, mlabel in enumerate(mode_labels):
        for cls_idx, cur_class in enumerate(classes):
            for diff_idx, diff_type in enumerate(difficulty_types):
                tp, fp, fn = counts[mode_idx, cls_idx, diff_idx]
                ret_dict[f'TP_{mlabel}_{cur_class}/{diff_type}'] = int(tp)
                ret_dict[f'FP_{mlabel}_{cur_class}/{diff_type}'] = int(fp)
                ret_dict[f'FN_{mlabel}_{cur_class}/{diff_type}'] = int(fn)

    # ── build result string (3D table / BEV table, then the counts) ──
    ret_str = _build_table_str(AP, mAP, classes, difficulty_types,
                               num_difficulties, num_pr_points, mode_labels)
    ret_str += _build_count_table_str(counts, classes, difficulty_types, mode_labels, count_score_thresh, iou_thresholds)

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


def _build_count_table_str(counts, classes, difficulty_types, mode_labels, score_thresh, iou_thresholds):
    """TP / FP / FN, precision and recall (%) per bin, one table per IoU mode"""
    col_w = 12
    sep = '+' + '-' * (col_w + 2)
    header_sep = sep + sep * len(difficulty_types) + '+'
    lines = []
    for mode_idx, mlabel in enumerate(mode_labels):
        ious = ', '.join('%s %.2f' % (c, iou_thresholds[c]) for c in classes)
        lines.append('\n[ %s counts, score >= %.2f, IoU > %s ]' % (mlabel, score_thresh, ious))
        lines.append(header_sep)
        hdr = '| {:<{w}} |'.format('', w=col_w)
        for dt in difficulty_types:
            hdr += ' {:<{w}} |'.format(dt, w=col_w)
        lines.append(hdr)
        lines.append(header_sep)
        for cls_idx, cur_class in enumerate(classes):
            c = counts[mode_idx, cls_idx]
            prec = [100.0 * tp / (tp + fp) if tp + fp > 0 else float('nan') for tp, fp, fn in c]
            rec = [100.0 * tp / (tp + fn) if tp + fn > 0 else float('nan') for tp, fp, fn in c]
            for name, vals, fmt in (('TP', c[:, 0], '{:<{w}d}'), ('FP', c[:, 1], '{:<{w}d}'), ('FN', c[:, 2], '{:<{w}d}'),
                                    ('Precision %', prec, '{:<{w}.2f}'), ('Recall %', rec, '{:<{w}.2f}')):
                label = name if len(classes) == 1 else '%s %s' % (cur_class, name)
                row = '| {:<{w}} |'.format(label, w=col_w)
                for v in vals:
                    row += (' ' + fmt + ' |').format(int(v) if 'd' in fmt else v, w=col_w)
                lines.append(row)
            lines.append(header_sep)
    return '\n'.join(lines) + '\n'


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

    if 'ignore' in gt_anno:  # boxes that are not labels (LABEL_FILTER): don't care
        gt_flag[(gt_flag == 0) & np.asarray(gt_anno['ignore'], dtype=bool)] = 1

    return gt_flag, pred_flag