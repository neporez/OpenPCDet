"""
Which annotated boxes count as labels.

A fixed minimum number of points is unfair to boats of very different sizes (8 m to 124 m): a small boat far away is
fully visible with 15 points, while a 100 m ship with 50 points may show only one end. So a box is judged against the
number of points it would get if fully visible:

    expected points  = C * visible_area / distance^2
    visible_area     = length * height * |sin(aspect)| + width * height * |cos(aspect)|
                       (aspect = line of sight - heading: 90 deg = side on, 0 deg = bow / stern on)
    visibility ratio = points / expected points

A box is a label if it has enough points to be recognisable whatever its size (points >= ABS_POINTS) or if enough of it
is visible (ratio >= RATIO). Occlusion is not modelled: an occluded box simply has a low ratio.

C (points per steradian) comes from the sensor: one point per beam, i.e. per azimuth step x elevation step,
    C = 1 / (dAz * dEl),  dAz = 2 pi / COLUMNS,  dEl = V_FOV / (CHANNELS - 1)   (uniform beam spacing)
so the ratio is the share of the theoretically possible returns. Real KRISO LiDAR: Ouster OS2-128 in 2048 x 10 mode,
22.5 deg vertical FOV (measured in the bins: 128 x 2048 rows, 0.176 deg / 0.170 deg steps) -> C = 105,413.
Even fully visible boats reach only part of it (box faces larger than the hull, dark / grazing surfaces): median ratio of
the real labels 0.23 (0.37 within 50 m, 0.08 beyond 200 m).

cfg (LABEL_FILTER in the dataset config): {RATIO: 0.1, ABS_POINTS: 50, SENSOR: {CHANNELS: 128, V_FOV_DEG: 22.5, COLUMNS: 2048}}
(a fixed C: ... instead of SENSOR is also accepted)
"""
import numpy as np


def sensor_c(sensor):
    """points per steradian of a rotating LiDAR with uniformly spaced beams"""
    d_az = 2 * np.pi / sensor['COLUMNS']
    d_el = np.radians(sensor['V_FOV_DEG']) / (sensor['CHANNELS'] - 1)
    return float(1.0 / (d_az * d_el))


def get_c(cfg):
    """C of a LABEL_FILTER cfg: computed from SENSOR, or the fixed value C"""
    if cfg.get('SENSOR', None) is not None:
        return sensor_c(cfg['SENSOR'])
    return float(cfg['C'])


def visible_area(boxes):
    """boxes: (N, >=7) [x, y, z, dx, dy, dz, yaw, ...] -> (N,) area of the box faces seen from the sensor (m^2)"""
    boxes = np.asarray(boxes, dtype=np.float64)
    aspect = np.arctan2(boxes[:, 1], boxes[:, 0]) - boxes[:, 6]
    return boxes[:, 3] * boxes[:, 5] * np.abs(np.sin(aspect)) + boxes[:, 4] * boxes[:, 5] * np.abs(np.cos(aspect))


def expected_points(boxes, c):
    boxes = np.asarray(boxes, dtype=np.float64)
    dist = np.maximum(np.linalg.norm(boxes[:, :3], axis=1), 1e-3)
    return c * visible_area(boxes) / dist ** 2


def visibility_ratio(boxes, num_points, c):
    return np.asarray(num_points, dtype=np.float64) / np.maximum(expected_points(boxes, c), 1e-6)


def is_label(boxes, num_points, cfg):
    """(N,) bool: True for boxes that count as labels under cfg = {RATIO, ABS_POINTS, SENSOR or C}"""
    num_points = np.asarray(num_points, dtype=np.float64)
    if len(num_points) == 0:
        return np.zeros(0, dtype=bool)
    ok = visibility_ratio(boxes, num_points, get_c(cfg)) >= cfg['RATIO']
    if cfg.get('ABS_POINTS', None) is not None:
        ok |= num_points >= cfg['ABS_POINTS']
    return ok


def fit_c(boxes, num_points, point_cloud_range):
    """(analysis only) C such that the median visibility ratio of the boxes inside the range (with points) is 1"""
    boxes = np.asarray(boxes, dtype=np.float64)
    num_points = np.asarray(num_points, dtype=np.float64)
    inside = np.all((boxes[:, :3] >= point_cloud_range[:3]) & (boxes[:, :3] <= point_cloud_range[3:]), axis=1) & (num_points > 0)
    return float(np.exp(np.median(np.log(num_points[inside]) - np.log(expected_points(boxes[inside], 1.0)))))
