import numpy as np


def compute_split_parts(num_samples, num_parts):
    part_samples = num_samples // num_parts
    remain_samples = num_samples % num_parts
    if part_samples == 0:
        return [num_samples]
    if remain_samples == 0:
        return [part_samples] * num_parts
    else:
        return [part_samples] * num_parts + [remain_samples]


def overall_filter(boxes):
    ignore = np.zeros(boxes.shape[0], dtype=bool)  # all false
    return ignore


def _distance_flag(dist, level):
    # level: 0 -> 0-100m, 1 -> 100-150m, 2 -> 0-150m, 3 -> 150m-inf
    if level == 0:
        return dist < 100
    elif level == 1:
        return (dist >= 100) & (dist < 150)
    elif level == 2:
        return dist < 150
    elif level == 3:
        return dist >= 150
    assert False, 'level < 4 for distance metric, found level %s' % (str(level))


def distance_filter(boxes, level):
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))
    ignore[_distance_flag(dist, level)] = False
    return ignore


def _length_flag(length, level):
    # level: 0 -> boats up to 40 m long, 1 -> longer than 40 m (box length dx = column 3)
    return length <= 40.0 if level == 0 else length > 40.0


def overall_distance_filter(boxes, level):
    # level 0 -> overall, 1..4 -> distance levels 0..3, 5..6 -> length levels 0..1 (len0-40m, len40m-inf).
    # Like the distance bins, gt boxes are binned by their own size and predictions by their predicted size;
    # boxes outside the bin are "don't care" (COCO-style size AP)
    ignore = np.ones(boxes.shape[0], dtype=bool)  # all true
    if level == 0:
        flag = np.ones(boxes.shape[0], dtype=bool)
    elif level <= 4:
        dist = np.sqrt(np.sum(boxes[:, 0:3] * boxes[:, 0:3], axis=1))
        flag = _distance_flag(dist, level - 1)
    else:
        flag = _length_flag(boxes[:, 3], level - 5)
    ignore[flag] = False
    return ignore
