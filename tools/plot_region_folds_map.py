"""Figure for the report: Bangeojin region folds (10 interleaved sectors) on an OpenStreetMap basemap (contextily, OSM needs a
User-Agent). Labels (world frame) coloured by fold, grey = 20 m buffer, ego trajectories, sector rays. Output: --out <png>.
python plot_region_folds_map.py [--out ../output/region_folds_map.png]"""
import argparse, csv, json, pickle, sys, numpy as np, yaml
sys.path.insert(0, '/home/rain/OpenPCDet')
from pcdet.datasets.kriso import region_utils
from pcdet.utils import visibility_utils
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import contextily as ctx
D = '/home/rain/OpenPCDet/data/kriso_ulsan/'
lf = yaml.safe_load(open('/home/rain/OpenPCDet/tools/cfgs/dataset_configs/kriso_dataset.yaml'))['LABEL_FILTER']
reg = region_utils.load_regions(D + 'ImageSets/bangeojin_regions10.json'); folds = json.load(open(D + 'ImageSets/bangeojin_regions10.json'))['folds']; C = reg['centre']
lat0, lon0, R = 35.5045033, 129.3653983, 6378137.0
_align = json.load(open(D + 'ImageSets/map_alignment.json')) if __import__('os').path.exists(D + 'ImageSets/map_alignment.json') else None   # figure-only alignment from adjust_region_map.py
def to_lonlat(xy):
    xy = np.asarray(xy, float)
    if _align:
        c0 = np.asarray(_align['rotation_centre']); t = np.radians(_align['dth']); c, s = np.cos(t), np.sin(t); v = xy[:, :2] - c0
        xy = np.c_[v[:, 0] * c - v[:, 1] * s, v[:, 0] * s + v[:, 1] * c] + c0 + [_align['dx'], _align['dy']]
    return np.c_[lon0 + np.degrees(xy[:, 0] / (R * np.cos(np.radians(lat0)))), lat0 + np.degrees(xy[:, 1] / R)]
poses = {(r['sequence_id'], r['frame_id']): [float(r['x']), float(r['y']), float(r['yaw'])] for r in csv.DictReader(open(D + 'ImageSets/bangeojin_world_poses_v2.csv'))}
infos = [i for i in pickle.load(open(D + 'kriso_ulsan_infos_real.pkl', 'rb')) if (i['sequence_id'], i['frame_id']) in poses]
L = []
for i in infos:
    a = i['annos']
    if len(a['name']) == 0: continue
    lab = visibility_utils.is_label(a['boxes_3d'], a['num_points_in_gt'], lf); L.append(region_utils.to_world(a['boxes_3d'][:, :2], poses[(i['sequence_id'], i['frame_id'])])[lab])
L = np.concatenate(L); sec = region_utils.sector(reg, L); f_of = {k: f for f, secs in enumerate(folds) for k in secs}
buf = np.zeros(len(L), bool)
for k in range(len(reg['boundaries'])): buf |= region_utils.near_ray(reg, L, k)
fig, ax = plt.subplots(figsize=(14, 14)); cols = plt.cm.tab10(np.arange(5)); ll = to_lonlat(L)
for f in range(5):
    m = np.isin(sec, folds[f]) & ~buf; ax.scatter(ll[m, 0], ll[m, 1], s=6, color=cols[f], label='fold %d = sectors %s' % (f, folds[f]), zorder=3)
ax.scatter(ll[buf, 0], ll[buf, 1], s=6, color='0.45', label='buffer (20 m)', zorder=3)
ego = to_lonlat(np.array(list(poses.values()))[:, :2]); ax.scatter(ego[:, 0], ego[:, 1], s=2, c='k', alpha=0.35, label='ego trajectory', zorder=2)
cl = to_lonlat(C[None])[0]; b = reg['boundaries']
for k, bb in enumerate(b):
    d = np.array([np.cos(np.radians(bb)), np.sin(np.radians(bb))]); seg = to_lonlat(np.array([C, C + 450 * d])); ax.plot(seg[:, 0], seg[:, 1], 'k--', lw=1.0, zorder=2)
    nxt = b[k + 1] if k + 1 < len(b) else b[0] + 360; mid = np.radians((bb + nxt) / 2); p = to_lonlat((C + 400 * np.array([np.cos(mid), np.sin(mid)]))[None])[0]
    ax.text(p[0], p[1], 'S%d / fold %d' % (k, f_of[k]), fontsize=11, ha='center', va='center', color=cols[f_of[k]], weight='bold', zorder=4, bbox=dict(facecolor='white', alpha=0.7, lw=0))
ax.plot(cl[0], cl[1], 'r*', ms=14, zorder=4)
ext = to_lonlat(np.array([C - 470, C + 470])); ax.set_xlim(ext[0, 0], ext[1, 0]); ax.set_ylim(ext[0, 1], ext[1, 1])
ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=17, attribution='© OpenStreetMap contributors', headers={'User-Agent': 'kriso-boat-detection-report-figure/1.0 (academic; contact: chatssb1234@gmail.com)'})
ax.set_aspect(1 / np.cos(np.radians(lat0))); ax.set_title('Bangeojin port: region 5-fold (10 sectors, fold k = sectors k and k+5); points = boat labels (world frame)', fontsize=12); ax.legend(loc='lower left', fontsize=10); ax.set_xlabel('longitude'); ax.set_ylabel('latitude')
_ap = argparse.ArgumentParser(); _ap.add_argument('--out', default='/home/rain/OpenPCDet/output/region_folds_map.png'); out = _ap.parse_args().out or '/home/rain/OpenPCDet/output/region_folds_map.png'; plt.tight_layout(); plt.savefig(out, dpi=90); print('saved', out)
