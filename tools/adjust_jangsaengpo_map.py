"""
Interactive alignment of the Jangsaengpo label map (kriso_world scenes, GNSS reference frame) over the OpenStreetMap basemap,
for the report figure only (the fold split lives in the reference frame and is not changed here).

Shows: ego paths of the 4 Jangsaengpo scenes (1105 006 / 007, 1106 000 / 008), static labels (size ~ length) coloured by
fold, the 10 strip boundaries (perpendicular to the harbour axis, from ImageSets/jangsaengpo_strips10.json), grey = buffer.
keys: arrows = move 1 m (shift + arrows = 10 m), Q / E = rotate -/+ 0.2 deg about the strip centre (shift: 1 deg),
      R = reset, S = save ImageSets/map_alignment_jangsaengpo.json + output/jangsaengpo_folds_map.png, Esc = quit

python adjust_jangsaengpo_map.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
SAVE_ONLY = '--save' in sys.argv            # headless: just write the png with the saved alignment
if SAVE_ONLY:
    matplotlib.use('Agg')
for backend in (() if SAVE_ONLY else ('QtAgg', 'TkAgg', 'GTK3Agg')):
    try:
        matplotlib.use(backend); break
    except Exception:
        continue
import matplotlib.pyplot as plt
import contextily as ctx
for _k in ('keymap.quit', 'keymap.quit_all', 'keymap.save', 'keymap.home', 'keymap.back', 'keymap.forward', 'keymap.fullscreen',
           'keymap.grid', 'keymap.grid_minor', 'keymap.pan', 'keymap.zoom', 'keymap.xscale', 'keymap.yscale', 'keymap.copy'):
    if _k in matplotlib.rcParams:
        matplotlib.rcParams[_k] = []

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / 'data' / 'kriso_ulsan'
KW = Path('/media/rain/Extreme SSD/kriso_world')
STRIPS = D / 'ImageSets' / 'jangsaengpo_strips10.json'
ALIGN = D / 'ImageSets' / 'map_alignment_jangsaengpo.json'
SCENES = [('20261105_165321_jangsaeng', 6), ('20261105_165321_jangsaeng', 7), ('20261106_120833_jangsaeng', 0), ('20261106_120833_jangsaeng', 8)]
LAT0, LON0, R = 35.5045033, 129.3653983, 6378137.0
HEADERS = {'User-Agent': 'kriso-boat-detection-report-figure/1.0 (academic; contact: chatssb1234@gmail.com)'}


def to_lonlat(xy):
    xy = np.asarray(xy, float)
    return np.c_[LON0 + np.degrees(xy[:, 0] / (R * np.cos(np.radians(LAT0)))), LAT0 + np.degrees(xy[:, 1] / R)]


ALIGN_SCENES = D / 'ImageSets' / 'jangsaengpo_scene_align.json'


def load_scenes():
    """ego paths + static labels in the reference frame, with the per-scene 2D alignment (jangsaengpo_scene_align.json) applied"""
    egos, labels, lens = [], [], []
    al = json.load(open(ALIGN_SCENES))['scenes'] if ALIGN_SCENES.exists() else {}
    for bag, s in SCENES:
        sd = KW / bag / ('scene_%03d' % s)
        o = np.array(json.load(open(sd / 'scene.json'))['origin_in_reference_frame'][:2], float)
        sa = al.get('%s/scene_%03d' % (bag, s)); Rm = np.array(sa['R']) if sa else np.eye(2); t = np.array(sa['t']) if sa else np.zeros(2)
        egos.append((np.array([[float(r['x']), float(r['y'])] for r in csv.DictReader(open(sd / 'poses.csv'))]) + o) @ Rm.T + t)
        L = json.load(open(sd / 'labels.json'))
        for b in L['objects']:
            if b['kind'] == 'static':
                labels.append(Rm @ (np.array([b['pose']['x'], b['pose']['y']]) + o) + t); lens.append(b['size'][0])
    return egos, np.array(labels), np.array(lens)


def main():
    egos, L, lens = load_scenes()
    strips = json.load(open(STRIPS))
    C = np.array(strips['centre_reference_frame'], float); a = np.radians(strips['axis_deg']); d = np.array([np.cos(a), np.sin(a)]); n = np.array([-d[1], d[0]])
    bounds = np.array(strips['boundaries_m'], float); folds = strips['folds']; buf = strips['buffer_m']
    proj = (L - C) @ d
    strip = np.clip(np.searchsorted(bounds, proj, side='right') - 1, 0, len(bounds) - 2)
    inbuf = np.min(np.abs(proj[:, None] - bounds[None, 1:-1]), axis=1) < buf
    f_of = {k: f for f, ks in enumerate(folds) for k in ks}
    fold = np.array([f_of[k] for k in strip])
    half = 330.0
    lines = [np.array([C + d * b - n * half, C + d * b + n * half]) for b in bounds]
    st = {'dx': 0.0, 'dy': 0.0, 'dth': 0.0}
    if ALIGN.exists():
        st.update({k: float(v) for k, v in json.load(open(ALIGN)).items() if k in st})

    def tf(xy):
        t = np.radians(st['dth']); c, s = np.cos(t), np.sin(t); v = np.asarray(xy, float) - C
        return np.c_[v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c] + C + [st['dx'], st['dy']]

    fig, ax = plt.subplots(figsize=(16, 10)); cols = plt.cm.tab10(np.arange(5)); arts = {}
    for f in range(5):
        m = (fold == f) & ~inbuf
        arts[f] = ax.scatter(*to_lonlat(tf(L[m])).T, s=np.clip(lens[m], 6, 70), color=cols[f], edgecolor='k', lw=0.3, label='fold %d = strips %s' % (f, folds[f]), zorder=3)
    arts['buf'] = ax.scatter(*to_lonlat(tf(L[inbuf])).T, s=np.clip(lens[inbuf], 6, 70), color='0.45', edgecolor='k', lw=0.3, label='buffer (%g m)' % buf, zorder=3)
    arts['ego'] = [ax.plot(*to_lonlat(tf(e)).T, '-', color='k', lw=1.0, alpha=0.5, zorder=2)[0] for e in egos]
    arts['lines'] = [ax.plot(*to_lonlat(tf(ln)).T, 'k--', lw=0.9, zorder=2)[0] for ln in lines]
    allp = np.concatenate([L] + egos); lo, hi = allp.min(0) - 150, allp.max(0) + 150
    ext = to_lonlat(np.array([lo, hi])); ax.set_xlim(ext[0, 0], ext[1, 0]); ax.set_ylim(ext[0, 1], ext[1, 1])
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=17, attribution='© OpenStreetMap contributors', headers=HEADERS)
    ax.set_aspect(1 / np.cos(np.radians(LAT0))); ax.legend(loc='lower right', fontsize=9)

    def title():
        ax.set_title('Jangsaengpo | arrows: move 1 m (shift 10 m) | Q/E: rotate 0.2 deg (shift 1 deg) | R reset | S save | Esc quit\n'
                     'offset east %+.1f m  north %+.1f m  rotation %+.2f deg' % (st['dx'], st['dy'], st['dth']), fontsize=11)

    def redraw():
        for f in range(5):
            m = (fold == f) & ~inbuf; arts[f].set_offsets(to_lonlat(tf(L[m])))
        arts['buf'].set_offsets(to_lonlat(tf(L[inbuf])))
        for art, e in zip(arts['ego'], egos):
            p = to_lonlat(tf(e)); art.set_data(p[:, 0], p[:, 1])
        for art, ln in zip(arts['lines'], lines):
            p = to_lonlat(tf(ln)); art.set_data(p[:, 0], p[:, 1])
        title(); fig.canvas.draw_idle()

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
            json.dump({'dx_m': st['dx'], 'dy_m': st['dy'], 'dtheta_deg': st['dth'], 'rotation_centre': list(map(float, C)), 'dx': st['dx'], 'dy': st['dy'], 'dth': st['dth'],
                       'note': 'figure-only alignment of the Jangsaengpo GNSS reference frame onto the OSM basemap (rotation about the strip centre, then translation)'}, open(ALIGN, 'w'), indent=1)
            out = ROOT / 'output' / 'jangsaengpo_folds_map.png'; fig.savefig(out, dpi=90); print('saved %s and %s' % (ALIGN, out)); return
        elif base == 'escape': plt.close(fig); return
        else: return
        redraw()
    title(); plt.tight_layout()
    if SAVE_ONLY:
        ax.set_title('Jangsaengpo region folds: 10 equal-width strips, fold k = strips {k, k+5}, grey = 20 m buffer', fontsize=12)
        out = ROOT / 'output' / 'jangsaengpo_folds_map.png'; fig.savefig(out, dpi=90); print('saved', out); return
    fig.canvas.mpl_connect('key_press_event', on_key); plt.show()


if __name__ == '__main__':
    main()
