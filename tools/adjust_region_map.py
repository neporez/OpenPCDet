"""
Interactive alignment of the Bangeojin label map (world frame) over the OpenStreetMap basemap, for the report figure only
(the fold split is unaffected: it lives in the GNSS reference frame).

keys: arrows = move 1 m (shift + arrows = 10 m), Q / E = rotate -/+ 0.2 deg about the sector centre (shift: 1 deg),
      R = reset, S = save the transform to ImageSets/map_alignment.json + the figure output/region_folds_map.png, Esc = quit
The transform (dx, dy in metres, dtheta in degrees, rotation about the sector centre) is applied to every world point
before the lon/lat conversion; plot_region_folds_map.py reads the json when it exists.

python adjust_region_map.py
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

import _init_path  # noqa
from pcdet.datasets.kriso import region_utils
from pcdet.utils import visibility_utils

import matplotlib
for backend in ('QtAgg', 'TkAgg', 'GTK3Agg'):
    try:
        matplotlib.use(backend); break
    except Exception:
        continue
import matplotlib.pyplot as plt
import contextily as ctx
# matplotlib's own key bindings collide with ours (q = quit, s = save, r = home, left/right = back/forward ...): disable them
for _k in ('keymap.quit', 'keymap.quit_all', 'keymap.save', 'keymap.home', 'keymap.back', 'keymap.forward', 'keymap.fullscreen',
           'keymap.grid', 'keymap.grid_minor', 'keymap.pan', 'keymap.zoom', 'keymap.xscale', 'keymap.yscale', 'keymap.copy'):
    if _k in matplotlib.rcParams:
        matplotlib.rcParams[_k] = []

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data' / 'kriso_ulsan'
ALIGN = D / 'ImageSets' / 'map_alignment.json'
LAT0, LON0, R = 35.5045033, 129.3653983, 6378137.0
HEADERS = {'User-Agent': 'kriso-boat-detection-report-figure/1.0 (academic; contact: chatssb1234@gmail.com)'}


def to_lonlat(xy):
    xy = np.asarray(xy, float)
    return np.c_[LON0 + np.degrees(xy[:, 0] / (R * np.cos(np.radians(LAT0)))), LAT0 + np.degrees(xy[:, 1] / R)]


def main():
    lf = yaml.safe_load(open(ROOT / 'tools/cfgs/dataset_configs/kriso_dataset.yaml'))['LABEL_FILTER']
    rj = json.load(open(D / 'ImageSets/bangeojin_regions10.json')); reg = region_utils.load_regions(D / 'ImageSets/bangeojin_regions10.json')
    folds = rj['folds']; C = reg['centre']
    poses = {(r['sequence_id'], r['frame_id']): [float(r['x']), float(r['y']), float(r['yaw'])] for r in csv.DictReader(open(D / 'ImageSets/bangeojin_world_poses_v2.csv'))}
    infos = [i for i in pickle.load(open(D / 'kriso_ulsan_infos_real.pkl', 'rb')) if (i['sequence_id'], i['frame_id']) in poses]
    L = []
    for i in infos:
        a = i['annos']
        if len(a['name']) == 0:
            continue
        lab = visibility_utils.is_label(a['boxes_3d'], a['num_points_in_gt'], lf)
        L.append(region_utils.to_world(a['boxes_3d'][:, :2], poses[(i['sequence_id'], i['frame_id'])])[lab])
    L = np.concatenate(L); sec = region_utils.sector(reg, L); f_of = {k: f for f, secs in enumerate(folds) for k in secs}
    buf = np.zeros(len(L), bool)
    for k in range(len(reg['boundaries'])):
        buf |= region_utils.near_ray(reg, L, k)
    ego = np.array(list(poses.values()))[:, :2]
    rays = [np.array([C, C + 450 * np.array([np.cos(np.radians(b)), np.sin(np.radians(b))])]) for b in reg['boundaries']]
    st = {'dx': 0.0, 'dy': 0.0, 'dth': 0.0}
    if ALIGN.exists():
        st.update({k: float(v) for k, v in json.load(open(ALIGN)).items() if k in st})

    def tf(xy):
        t = np.radians(st['dth']); c, s = np.cos(t), np.sin(t); v = np.asarray(xy, float) - C
        return np.c_[v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c] + C + [st['dx'], st['dy']]

    fig, ax = plt.subplots(figsize=(13, 13)); cols = plt.cm.tab10(np.arange(5))
    arts = {}
    for f in range(5):
        m = np.isin(sec, folds[f]) & ~buf; arts[f] = ax.scatter(*to_lonlat(tf(L[m])).T, s=6, color=cols[f], label='fold %d = sectors %s' % (f, folds[f]), zorder=3)
    arts['buf'] = ax.scatter(*to_lonlat(tf(L[buf])).T, s=6, color='0.45', label='buffer (20 m)', zorder=3)
    arts['ego'] = ax.scatter(*to_lonlat(tf(ego)).T, s=2, c='k', alpha=0.35, label='ego trajectory', zorder=2)
    arts['rays'] = [ax.plot(*to_lonlat(tf(r)).T, 'k--', lw=1.0, zorder=2)[0] for r in rays]
    arts['centre'] = ax.plot(*to_lonlat(tf(C[None])).T, 'r*', ms=14, zorder=4)[0]
    ext = to_lonlat(np.array([C - 470, C + 470])); ax.set_xlim(ext[0, 0], ext[1, 0]); ax.set_ylim(ext[0, 1], ext[1, 1])
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=17, attribution='© OpenStreetMap contributors', headers=HEADERS)
    ax.set_aspect(1 / np.cos(np.radians(LAT0))); ax.legend(loc='lower left', fontsize=9)

    def title():
        ax.set_title('arrows: move 1 m (shift 10 m) | Q/E: rotate 0.2 deg (shift 1 deg) | R reset | S save | Esc quit\n'
                     'offset east %+.1f m  north %+.1f m  rotation %+.2f deg' % (st['dx'], st['dy'], st['dth']), fontsize=11)

    def redraw():
        for f in range(5):
            m = np.isin(sec, folds[f]) & ~buf; arts[f].set_offsets(to_lonlat(tf(L[m])))
        arts['buf'].set_offsets(to_lonlat(tf(L[buf]))); arts['ego'].set_offsets(to_lonlat(tf(ego)))
        for ln, r in zip(arts['rays'], rays):
            p = to_lonlat(tf(r)); ln.set_data(p[:, 0], p[:, 1])
        p = to_lonlat(tf(C[None])); arts['centre'].set_data(p[:, 0], p[:, 1]); title(); fig.canvas.draw_idle()

    def on_key(ev):
        k = ev.key or ''; step = 10.0 if 'shift' in k else 1.0; rot = 1.0 if 'shift' in k else 0.2; base = k.replace('shift+', '')
        if base == 'left': st['dx'] -= step
        elif base == 'right': st['dx'] += step
        elif base == 'up': st['dy'] += step
        elif base == 'down': st['dy'] -= step
        elif base in ('q', 'Q'): st['dth'] -= rot
        elif base in ('e', 'E'): st['dth'] += rot
        elif base in ('r', 'R'): st.update(dx=0.0, dy=0.0, dth=0.0)
        elif base in ('s', 'S'):
            json.dump({'dx_m': st['dx'], 'dy_m': st['dy'], 'dtheta_deg': st['dth'], 'rotation_centre': list(C), 'dx': st['dx'], 'dy': st['dy'], 'dth': st['dth'],
                       'note': 'figure-only alignment of the GNSS reference frame onto the OSM basemap (rotation about the sector centre, then translation)'}, open(ALIGN, 'w'), indent=1)
            out = ROOT / 'output' / 'region_folds_map.png'; fig.savefig(out, dpi=90); print('saved %s and %s' % (ALIGN, out)); return
        elif base == 'escape': plt.close(fig); return
        else: return
        redraw()
    fig.canvas.mpl_connect('key_press_event', on_key); title(); plt.tight_layout(); plt.show()


if __name__ == '__main__':
    main()
