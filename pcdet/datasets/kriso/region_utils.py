"""
Spatial regions for the region-based k-fold (ImageSets/<regions>.json).

Every frame has a world pose (info['world_pose'] = [x, y, yaw] of the LiDAR in the GNSS reference frame, see
ImageSets/world_poses_v3.csv) and belongs to a SITE. A site geometry is one of
  * sectors: region k = angular sector boundaries_deg[k] .. boundaries_deg[k+1] (wrapping) around centre_reference_frame
             (0 deg = +x = east, 90 deg = +y = north); buffer = within buffer_m of a boundary ray        (Bangeojin: ego loop)
  * strips : region k = boundaries_m[k] .. boundaries_m[k+1] along the axis (axis_deg) through centre_reference_frame;
             buffer = within buffer_m of an interior boundary line                                        (Jangsaengpo: straight channel)
A regions json is either one geometry (old files: bangeojin_regions*.json) or {"buffer_m", "sites": {name: geometry}}
(regions_v3.json). With several sites the per-frame info['region']['site'] selects the geometry.

For fold f with regions S_f of a site:
  excluded(S_f)  = inside a region of S_f, or within buffer_m of one of its boundaries  -> cut from the training frames
  val_zone(S_f)  = inside a region of S_f and not within buffer_m of its boundaries     -> evaluated in the val frames
  train_zone     = not excluded                                                           -> evaluated in train_eval
Objects are assigned by their box centre, points by their own position.
"""
import json

import numpy as np


def _prep(g):
    g = dict(g)
    g['centre'] = np.asarray(g['centre_reference_frame'], dtype=np.float64)
    g['type'] = g.get('type', 'sectors')
    if g['type'] == 'sectors':
        g['boundaries'] = [float(b) % 360 for b in g['boundaries_deg']]
        assert g['boundaries'] == sorted(g['boundaries']), 'boundaries_deg must be increasing'
        g['n'] = len(g['boundaries'])
    elif g['type'] == 'strips':
        g['boundaries'] = [float(b) for b in g['boundaries_m']]
        assert g['boundaries'] == sorted(g['boundaries']), 'boundaries_m must be increasing'
        a = np.radians(float(g['axis_deg']))
        g['axis'] = np.array([np.cos(a), np.sin(a)])
        g['n'] = len(g['boundaries']) - 1
    else:
        raise ValueError('unknown region type %s' % g['type'])
    return g


def load_regions(path):
    r = json.load(open(path))
    if 'sites' in r:
        buf = float(r.get('buffer_m', 20.0))
        sites = {}
        for name, g in r['sites'].items():
            g = dict(g); g.setdefault('buffer_m', buf); sites[name] = _prep(g)
        return {'multi': True, 'sites': sites, 'buffer_m': buf, 'folds': r.get('folds')}
    return _prep(r)


def geom(regions, region_info=None):
    """the site geometry for a frame (multi-site json: region_info['site'])"""
    if regions.get('multi'):
        site = (region_info or {}).get('site')
        if site is None:
            raise KeyError("multi-site regions need info['region']['site']")
        return regions['sites'][site]
    return regions


def to_world(xy, pose):
    """xy: (N, >=2) in the LiDAR frame, pose = [x, y, yaw] -> (N, 2) world xy"""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    xy = np.asarray(xy, dtype=np.float64)[:, :2]
    return np.c_[xy[:, 0] * c - xy[:, 1] * s + pose[0], xy[:, 0] * s + xy[:, 1] * c + pose[1]]


def region_index(g, wxy):
    """(N,) index of the region (sector / strip) of each world point"""
    v = np.asarray(wxy, dtype=np.float64) - g['centre']
    if g['type'] == 'sectors':
        ang = (np.degrees(np.arctan2(v[:, 1], v[:, 0])) + 360) % 360
        idx = np.searchsorted(g['boundaries'], ang, side='right') - 1          # ang < b[0] -> -1 -> last sector (wraps)
        return np.where(idx < 0, g['n'] - 1, idx)
    proj = v @ g['axis']
    return np.clip(np.searchsorted(g['boundaries'], proj, side='right') - 1, 0, g['n'] - 1)


sector = region_index   # old name


def near_boundary(g, wxy, k):
    """(N,) True within buffer_m of boundary k (sectors: ray k, beyond the centre the distance to the centre; strips: line k;
    the two open ends of a strip geometry have no buffer)"""
    v = np.asarray(wxy, dtype=np.float64) - g['centre']
    if g['type'] == 'sectors':
        a = np.radians(g['boundaries'][k % g['n']])
        d = np.array([np.cos(a), np.sin(a)])
        along = v @ d
        perp = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
        return np.where(along > 0, perp, np.linalg.norm(v, axis=1)) < g['buffer_m']
    if k <= 0 or k >= len(g['boundaries']) - 1:
        return np.zeros(len(v), dtype=bool)
    return np.abs(v @ g['axis'] - g['boundaries'][k]) < g['buffer_m']


near_ray = near_boundary   # old name


def _bounds_of(g, k):
    """indices of the two boundaries of region k"""
    return (k, (k + 1) % g['n']) if g['type'] == 'sectors' else (k, k + 1)


def excluded(g, wxy, k):
    b0, b1 = _bounds_of(g, k)
    return (region_index(g, wxy) == k) | near_boundary(g, wxy, b0) | near_boundary(g, wxy, b1)


def val_zone(g, wxy, k):
    b0, b1 = _bounds_of(g, k)
    return (region_index(g, wxy) == k) & ~near_boundary(g, wxy, b0) & ~near_boundary(g, wxy, b1)


def excluded_any(g, wxy, regions_out):
    out = np.zeros(len(wxy), dtype=bool)
    for k in regions_out:
        out |= excluded(g, wxy, k)
    return out


def val_zone_any(g, wxy, regions_in):
    out = np.zeros(len(wxy), dtype=bool)
    for k in regions_in:
        out |= val_zone(g, wxy, k)
    return out


def train_excluded(regions, wxy, region_info):
    """what a training frame loses: the union of excluded(k) over region_info['exclude'] (default: [fold])"""
    g = geom(regions, region_info)
    return excluded_any(g, wxy, region_info.get('exclude', [region_info['fold']]))


def zone_mask(regions, wxy, fold, mode, region_info=None):
    """mode 'val' -> val zone of the fold's regions (region_info['sectors'], default [fold]), 'train' -> not train_excluded"""
    region_info = region_info if region_info is not None else {'fold': fold}
    g = geom(regions, region_info)
    if mode == 'val':
        return val_zone_any(g, wxy, region_info.get('sectors', [fold]))
    return ~train_excluded(regions, wxy, region_info)
