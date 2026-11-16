"""
Interactive tool to choose the spatial regions of Bangeojin port for a region-based train / val split.

Regions are angular sectors around the harbour centre (centre of the ego paths), separated by boundary rays.
Boats (labelled boxes in world coordinates) are coloured by their region; boxes closer than the buffer to a boundary ray
are grey (they would be dropped from both training and evaluation of the neighbouring folds).

sliders:  boundary angles (0 = east, 90 = north), buffer (m); the number of regions = number of --cuts values (2-6)
panel:    per region its share of the boxes (1105 / 0609) and, if it is the validation fold, the share left for training
          (everything outside it and outside the buffers of its two boundaries)
keys:     S = save the boundaries to --out (json), Q = quit

python region_split_tool.py --data <region_tool_data.npz> --out ../data/kriso_ulsan/ImageSets/bangeojin_regions.json
"""
import argparse
import json

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons, Slider

COLS = ['tab:red', 'tab:blue', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='npz with M2 (map xy), B (boxes x, y, length, ...), E (ego xy), C (centre), lab (boat clusters)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cuts', type=float, nargs='+', default=[35, 95, 140, 180, 285], help='2-6 boundary angles (deg)')
    args = ap.parse_args()
    z = np.load(args.data)
    M2, B, E, C, lab = z['M2'], z['B'], z['E'], z['C'], z['lab']
    ang = (np.degrees(np.arctan2(B[:, 1] - C[1], B[:, 0] - C[0])) + 360) % 360

    fig = plt.figure(figsize=(15, 11))
    ax = fig.add_axes([0.05, 0.25, 0.62, 0.72])
    tax = fig.add_axes([0.70, 0.25, 0.28, 0.72]); tax.axis('off')
    ax.scatter(M2[::3, 0], M2[::3, 1], s=0.05, c='0.65')
    ax.plot(E[:, 0], E[:, 1], '.', ms=1, c='k', alpha=0.4)
    src = B[:, 3].astype(int) if B.shape[1] > 3 else np.zeros(len(B), int)   # 0 / 1 = 1105 cw / ccw, 2 = 0609
    is09 = src == 2
    sc = ax.scatter(B[~is09, 0], B[~is09, 1], s=2, marker='o')
    sc09 = ax.scatter(B[is09, 0], B[is09, 1], s=6, marker='^')
    if (src == 2).any():
        e09 = E[:, 2] == 2 if E.shape[1] > 2 else np.zeros(len(E), bool)
        ax.plot(E[e09, 0], E[e09, 1], '.', ms=1, c='tab:orange', alpha=0.5)
    N = len(args.cuts); assert 2 <= N <= 6
    rays = [ax.plot([], [], 'k--', lw=1.2)[0] for _ in range(N)]
    labels = [ax.text(0, 0, '', fontsize=12, weight='bold', ha='center') for _ in range(N)]
    ax.set_aspect('equal'); ax.set_xlim(-480, 480); ax.set_ylim(-400, 480); ax.grid(alpha=0.3)
    ax.set_xlabel('east (m)'); ax.set_ylabel('north (m)')
    ax.set_title('Bangeojin regions (S = save, Q = quit)')

    sl = [Slider(fig.add_axes([0.08, 0.205 - 0.03 * i, 0.55, 0.022]), 'boundary %d' % (i + 1), 0, 360, valinit=args.cuts[i], valstep=1)
          for i in range(N)]
    sb = Slider(fig.add_axes([0.08, 0.02, 0.55, 0.022]), 'buffer (m)', 0, 60, valinit=20, valstep=1)
    cbs = CheckButtons(fig.add_axes([0.83, 0.03, 0.15, 0.10]), ['show 1105', 'show 0609'], [True, True])
    state = {}

    def ray_dist(xy, a_deg):
        a = np.radians(a_deg); d = np.array([np.cos(a), np.sin(a)]); v = xy - C
        along = v @ d
        perp = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
        return np.where(along > 0, perp, np.linalg.norm(v, axis=1))

    def update(_=None):
        n = N
        cuts = sorted(s.val % 360 for s in sl)
        buf = sb.val
        region = np.zeros(len(B), int)
        edges = cuts + [cuts[0] + 360]
        for k in range(n):
            lo, hi = edges[k], edges[k + 1]
            region[((ang - lo) % 360) < (hi - lo)] = k
        near = np.array([ray_dist(B[:, :2], c) < buf for c in cuts])   # near boundary k (boundary k = start of region k)
        dropped = near.any(axis=0)
        colors = np.array([matplotlib.colors.to_rgba(COLS[r]) for r in region])
        colors[dropped] = (0.5, 0.5, 0.5, 0.5)
        vis05, vis09 = cbs.get_status()
        sc.set_facecolors(colors[~is09]); sc.set_edgecolors(colors[~is09]); sc.set_visible(vis05)
        sc09.set_facecolors(colors[is09]); sc09.set_edgecolors(colors[is09]); sc09.set_visible(vis09)
        lines = ['%d regions, buffer %d m\n' % (n, buf)]
        for k in range(N):
            if True:
                a = np.radians(cuts[k]); rays[k].set_data([C[0], C[0] + 450 * np.cos(a)], [C[1], C[1] + 450 * np.sin(a)])
                inr = (region == k) & ~dropped
                n05 = (inr & ~is09).sum(); n09 = (inr & is09).sum()
                val = (region == k) & ~near[k] & ~near[(k + 1) % n]          # evaluated boxes when region k is the val fold
                train = (region != k) & ~near[k] & ~near[(k + 1) % n]        # training boxes of that fold
                lo, hi = edges[k], edges[k + 1]
                mid = np.radians(lo + (hi - lo) / 2)
                labels[k].set_position((C[0] + 320 * np.cos(mid), C[1] + 320 * np.sin(mid)))
                labels[k].set_text('R%d\n%.0f%%' % (k + 1, 100 * inr.sum() / len(B))); labels[k].set_color(COLS[k])
                lines.append('R%d %3d -> %3d deg: 1105 %4.1f%%, 0609 %4.1f%%\n   as val fold: train %4.1f%% / val %4.1f%%'
                             % (k + 1, lo % 360, hi % 360, 100 * n05 / max((~is09).sum(), 1), 100 * n09 / max(is09.sum(), 1),
                                100 * train.mean(), 100 * val.mean()))
        lines.append('\nnear any boundary (grey): %d boxes (%.1f%%)\n(only dropped in the folds whose val region\n touches that boundary)' % (dropped.sum(), 100 * dropped.mean()))
        tax.clear(); tax.axis('off'); tax.text(0, 1, '\n'.join(lines), va='top', family='monospace', fontsize=11)
        state.update(cuts=[float(c) for c in cuts], buffer=float(buf))
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key in ('s', 'S'):
            off = z['ref_offset'] if 'ref_offset' in z else np.zeros(2)
            json.dump({'centre_reference_frame': [float(C[0] + off[0]), float(C[1] + off[1])],
                       'frame': 'hemisphere gnss reference_frame (origin lat 35.5045033, lon 129.3653983); 0609 = tf, 1105 = scene pose + origin_in_reference_frame',
                       'boundaries_deg': state['cuts'], 'buffer_m': state['buffer'],
                       'note': 'angles around centre_reference_frame, 0 = +x (east), 90 = +y (north)'},
                      open(args.out, 'w'), indent=2)
            print('saved', args.out, state, flush=True)
        elif ev.key in ('q', 'Q'):
            plt.close(fig)

    for s in sl + [sb]:
        s.on_changed(update)
    cbs.on_clicked(update)
    fig.canvas.mpl_connect('key_press_event', on_key)
    update()
    plt.show()
    print('final', state, flush=True)


if __name__ == '__main__':
    main()
