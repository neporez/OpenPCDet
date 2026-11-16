"""
KRISO boat dataset (LiDAR, single class `boat`). Boxes are stored as [x, y, z, dx, dy, dz, yaw, roll, pitch]
(real data and the 7-DoF "yaw" simulation have roll = pitch = 0; the "buoyancy" simulation has roll / pitch).

  * BOX_DOF (7 | 9): 7 = yaw-only boxes (roll / pitch are dropped), 9 = full boxes (needs the `tilt` head).
  * The simulator writes R = Rx(roll) Ry(pitch) Rz(yaw); the infos store R = Rz(yaw) Ry(pitch) Rx(roll)
    (converted once when the infos are built, see utils/box_rotation_utils.py).
  * LABEL_FILTER decides which boxes are labels (utils/visibility_utils.py: points >= ABS_POINTS or visibility ratio >= RATIO).
    Other boxes are removed from training and, with EVAL_IGNORE_NON_LABELS, ignored ("don't care") in evaluation.
  * gt_sampling: the database holds every real box; DB_INFO_PATH may contain {train_split} (one dbinfos file per training
    split, e.g. per fold, so no validation object is pasted into training) and PREPARE filter_by_visibility: True uses LABEL_FILTER.
  * Any split name is allowed: the sequences of a split are listed in ImageSets/<split>.txt
    (one folder of data/ per line) and the infos are read from INFO_PATH[<split>].

Build infos:  python -m pcdet.datasets.kriso.kriso_dataset --cfg_file tools/cfgs/dataset_configs/kriso_dataset.yaml --splits <names>
Build folds + gt database:  python -m pcdet.datasets.kriso.make_folds
"""
import copy
import json
import pickle
from pathlib import Path

import numpy as np

from ..dataset import DatasetTemplate
from ...utils import box_rotation_utils, visibility_utils
from . import region_utils


class KrisoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        dataset_cfg = copy.deepcopy(dataset_cfg)  # the augmentor configs are filled in below
        label_filter = dataset_cfg.get('LABEL_FILTER', None)
        # TRAIN_LABEL_FILTER (optional): a different rule for the TRAINING labels and the gt_sampling database only;
        # evaluation always uses LABEL_FILTER, so runs with different training rules stay comparable (2026-11-11)
        train_label_filter = dataset_cfg.get('TRAIN_LABEL_FILTER', None) or label_filter
        for aug_cfg in dataset_cfg.get('DATA_AUGMENTOR', {}).get('AUG_CONFIG_LIST', []):
            # the augmentations need to know how to read the box columns beyond the 7th
            aug_cfg['BOX_DOF'] = int(dataset_cfg.get('BOX_DOF', 9))
            if aug_cfg.get('NAME') == 'gt_sampling':
                # one database index per training split (e.g. per fold), and the same label rule as the dataset
                aug_cfg['DB_INFO_PATH'] = [p.replace('{train_split}', dataset_cfg.DATA_SPLIT['train']) for p in aug_cfg['DB_INFO_PATH']]
                if aug_cfg.get('PREPARE', {}).get('filter_by_visibility', False) is True:
                    assert train_label_filter is not None, 'filter_by_visibility needs LABEL_FILTER in the dataset config'
                    aug_cfg['PREPARE']['filter_by_visibility'] = dict(train_label_filter)
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.box_dof = int(dataset_cfg.get('BOX_DOF', 9))
        assert self.box_dof in (7, 9), 'BOX_DOF must be 7 or 9, got %s' % self.box_dof
        self.label_filter = label_filter
        self.train_label_filter = train_label_filter
        # multi-frame accumulation (2026-11-12): ACCUMULATE {NUM_FRAMES: 3, POSES: scene_poses_1105.csv} -> get_lidar returns the
        # frame + its (NUM_FRAMES-1)/2 previous and next labelled frames of the same scene, transformed into the frame's sensor
        # coordinates with the ICP scene poses (lidar -> scene world: x y z + quaternion). Only ulsan_1105 frames have poses;
        # other sequences (0609, sim) stay single-frame. Applied in training AND evaluation (test-time accumulation).
        acc = dataset_cfg.get('ACCUMULATE', None)
        self.acc_frames = int(acc.get('NUM_FRAMES', 1)) if acc else 1
        self.acc_table = {}
        if self.acc_frames > 1:
            import csv
            with open(self.root_path / 'ImageSets' / acc['POSES']) as f:
                for r in csv.DictReader(f):
                    q = np.array([float(r['qw']), float(r['qx']), float(r['qy']), float(r['qz'])]); w, x, y, z = q / np.linalg.norm(q)
                    T = np.eye(4)
                    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                                 [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                                 [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
                    T[:3, 3] = [float(r['x']), float(r['y']), float(r['z'])]
                    self.acc_table[('ulsan_1105', r['frame_id'])] = (T, r['prev'], r['next'])
            if logger is not None:
                logger.info('KRISO multi-frame accumulation: %d frames (poses for %d ulsan_1105 frames)' % (self.acc_frames, len(self.acc_table)))
        if logger is not None and train_label_filter is not None and dict(train_label_filter) != dict(label_filter or {}):
            logger.info('KRISO training label rule differs from the evaluation rule: train %s | eval %s' % (dict(train_label_filter), dict(label_filter)))
        self.eval_ignore_non_labels = dataset_cfg.get('EVAL_IGNORE_NON_LABELS', False)
        self.eval_filter_gt_by_range = dataset_cfg.get('EVAL_FILTER_GT_BY_RANGE', True)
        # score 7-DoF predictions with the 6-DoF IoU as well: roll = pitch = 0 is appended to them, so a model that
        # only knows yaw is compared against the tilted gt boxes (otherwise the IoU falls back to yaw only)
        self.eval_force_6dof = dataset_cfg.get('EVAL_FORCE_6DOF', False)
        # > 0: gt boxes longer than this (m) are "don't care" in evaluation (cross-site test: jangsaeng's > 40 m ships
        # have no counterpart in the training site); <= 0: off
        self.eval_ignore_length_above = float(dataset_cfg.get('EVAL_IGNORE_LENGTH_ABOVE', -1.0))
        self.num_point_features = int(dataset_cfg.get('NUM_POINT_FEATURES', 3))
        # region-based folds (make_region_folds.py): infos with info['region'] = {'fold', 'mode'} and info['world_pose']
        region_file = dataset_cfg.get('REGION_FILE', None)
        self.regions = region_utils.load_regions(self.root_path / 'ImageSets' / region_file) if region_file else None

        self.split = dataset_cfg.DATA_SPLIT['train'] if training else dataset_cfg.DATA_SPLIT['test']
        self.sample_seq_list = self._read_split_list(self.split)

        self.once_infos = []
        self.include_data(self.split)

    def _read_split_list(self, split):
        split_file = self.root_path / 'ImageSets' / (split + '.txt')
        if not split_file.exists():
            return None
        return [x.strip() for x in open(split_file).readlines() if x.strip()]

    def include_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading KRISO dataset (split=%s, BOX_DOF=%d)' % (split, self.box_dof))
        infos = []
        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos.extend(pickle.load(f))
        infos = [info for info in infos if 'annos' in info]
        self.once_infos.extend(infos)
        if self.logger is not None:
            self.logger.info('Total samples for KRISO dataset: %d' % len(infos))

    def get_lidar(self, sequence_id, frame_id, accumulate=True):
        path = self.root_path / 'data' / sequence_id / 'lidar_roof' / ('%s.bin' % frame_id)
        pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        if not accumulate or self.acc_frames <= 1 or (sequence_id, frame_id) not in self.acc_table:
            return pts
        T_cur, prev, nxt = self.acc_table[(sequence_id, frame_id)]
        half = (self.acc_frames - 1) // 2
        nbs = []
        f = frame_id
        for _ in range(half):
            f = self.acc_table[(sequence_id, f)][1] if (sequence_id, f) in self.acc_table else ''
            if not f: break
            nbs.append(f)
        f = frame_id
        for _ in range(half):
            f = self.acc_table[(sequence_id, f)][2] if (sequence_id, f) in self.acc_table else ''
            if not f: break
            nbs.append(f)
        T_inv = np.linalg.inv(T_cur); out = [pts]
        for nb in nbs:
            q = np.fromfile(str(self.root_path / 'data' / sequence_id / 'lidar_roof' / ('%s.bin' % nb)), dtype=np.float32).reshape(-1, 4)
            q = q[np.isfinite(q).all(axis=1)]
            M = T_inv @ self.acc_table[(sequence_id, nb)][0]
            xyz = q[:, :3].astype(np.float64) @ M[:3, :3].T + M[:3, 3]
            out.append(np.c_[xyz.astype(np.float32), q[:, 3:4]])
        return np.concatenate(out, axis=0)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.once_infos) * self.total_epochs
        return len(self.once_infos)

    def _select_boxes(self, boxes):
        return boxes[:, :self.box_dof]

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.once_infos)

        info = copy.deepcopy(self.once_infos[index])
        points = self.get_lidar(info['sequence_id'], info['frame_id'])
        if self.training and info.get('meta_info', {}).get('point_keep_prob', False):
            # simulated sequence with a per-point keep probability in column 4 (tools/make_pkeep_sequence.py):
            # a fresh Bernoulli thinning every time the frame is loaded, so each epoch sees a different dropout
            points = points[np.random.random(len(points)) < points[:, 3]]
        points = points[:, :self.num_point_features]
        if 'region' in info and info['region']['mode'] == 'train':
            # region fold: the validation region (+ buffer) is cut out of every training frame (its boxes are already
            # removed from the infos); NaN rows stay (no position) and are dropped later by the range mask
            assert self.regions is not None, 'region infos need REGION_FILE in the dataset config'
            w = region_utils.to_world(np.nan_to_num(points[:, :2], nan=1e6), info['world_pose'])
            points = points[~region_utils.train_excluded(self.regions, w, info['region']) | ~np.isfinite(points[:, 0])]

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
            'sequence_id': info['sequence_id'],  # frame ids repeat across sequences (0609 / 1105)
            'epoch_total': getattr(self, 'epoch_total', 1),
            'epoch_current': getattr(self, 'epoch_cuurent', 0)
        }

        annos = info['annos']
        names, boxes = annos['name'], annos['boxes_3d']
        if self.training and self.train_label_filter is not None:
            keep = visibility_utils.is_label(boxes, annos['num_points_in_gt'], self.train_label_filter)
            names, boxes = names[keep], boxes[keep]
        input_dict.update({'gt_names': names, 'gt_boxes': self._select_boxes(boxes)})

        return self.prepare_data(data_dict=input_dict)

    # ------------------------------------------------------------------ infos
    def get_infos(self, num_workers=4, sample_seq_list=None):
        import concurrent.futures as futures
        root_path = self.root_path

        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            seq_infos = []
            with open(Path(root_path) / 'data' / seq_idx / ('%s.json' % seq_idx), 'r') as f:
                info_this_seq = json.load(f)
            meta_info = info_this_seq['meta_info']

            for frame in info_this_seq['frames']:
                frame_id = frame['frame_id']
                frame_dict = {'sequence_id': seq_idx, 'frame_id': frame_id, 'meta_info': meta_info}
                for key in ('scene_id', 'snap_idx'):
                    if key in frame:
                        frame_dict[key] = frame[key]

                annos = frame['annos']
                if len(annos['boxes_3d']) == 0:
                    print('skip empty frame', seq_idx, frame_id)
                    continue
                boxes_3d = np.array(annos['boxes_3d'], dtype=np.float64)
                if boxes_3d.shape[1] == 7:
                    boxes_3d = np.concatenate([boxes_3d, np.zeros((boxes_3d.shape[0], 2))], axis=1)
                assert boxes_3d.shape[1] == 9, 'expected 7 or 9 box values, got %d' % boxes_3d.shape[1]

                boxes_3d = box_rotation_utils.sim_boxes_to_std(boxes_3d)
                points4 = self.get_lidar(seq_idx, frame_id, accumulate=False)   # infos always count single-frame points
                points = points4[:, :3]
                if meta_info.get('point_keep_prob', False):
                    # keep probabilities in column 4: store the EXPECTED point count (sum of p inside the box), so the
                    # label rule judges the thinned density the training actually sees
                    inside = box_rotation_utils.points_in_boxes_9dof(points.astype(np.float64), boxes_3d)
                    num_points = np.rint(inside.astype(np.float64) @ points4[:, 3].astype(np.float64)).astype(np.int32)
                else:
                    num_points = box_rotation_utils.count_points_in_boxes_9dof(points, boxes_3d).astype(np.int32)
                frame_dict['annos'] = {
                    'name': np.array(annos['names']),
                    'boxes_3d': boxes_3d.astype(np.float32),
                    'num_points_in_gt': num_points,
                }
                seq_infos.append(frame_dict)
            return seq_infos

        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            future_to_seq = {executor.submit(process_single_sequence, seq): seq for seq in sample_seq_list}
            all_infos = []
            for future in futures.as_completed(future_to_seq):
                all_infos.extend(future.result())
        # as_completed order is arbitrary; keep the sequence order of the split file
        order = {seq: i for i, seq in enumerate(sample_seq_list)}
        all_infos.sort(key=lambda info: (order[info['sequence_id']], info['frame_id']))
        return all_infos

    # ------------------------------------------------------------- prediction
    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            if pred_scores.shape[0] == 0:
                return {'name': np.zeros(0), 'score': np.zeros(0), 'boxes_3d': np.zeros((0, self.box_dof))}
            return {'name': np.array(class_names)[pred_labels - 1], 'score': pred_scores, 'boxes_3d': pred_boxes}

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'sequence_id' in batch_dict:
                single_pred_dict['sequence_id'] = batch_dict['sequence_id'][index]
            annos.append(single_pred_dict)
        return annos

    # -------------------------------------------------------------- evaluation
    def _eval_gt_annos(self):
        """GT annos for evaluation: boxes outside the point cloud range are dropped (EVAL_FILTER_GT_BY_RANGE), boxes that are
        not labels under LABEL_FILTER are marked 'ignore' (EVAL_IGNORE_NON_LABELS), and so are boxes longer than
        EVAL_IGNORE_LENGTH_ABOVE: they count neither as missed boxes nor, when a prediction matches them, as false positives."""
        gt_annos = []
        for info in self.once_infos:
            annos = copy.deepcopy(info['annos'])
            keep = np.ones(len(annos['name']), dtype=bool)
            if self.eval_filter_gt_by_range:
                center = annos['boxes_3d'][:, :3]
                keep &= np.all((center >= self.point_cloud_range[:3]) & (center <= self.point_cloud_range[3:]), axis=1)
            for key in ('name', 'boxes_3d', 'num_points_in_gt'):
                annos[key] = annos[key][keep]
            if self.eval_ignore_non_labels and self.label_filter is not None:
                annos['ignore'] = ~visibility_utils.is_label(annos['boxes_3d'], annos['num_points_in_gt'], self.label_filter)
            if self.eval_ignore_length_above > 0:
                long_box = annos['boxes_3d'][:, 3] > self.eval_ignore_length_above
                annos['ignore'] = annos.get('ignore', np.zeros(len(long_box), dtype=bool)) | long_box
            if 'region' in info:  # region fold: only boxes whose centre lies in the evaluated zone count
                zone = self._region_zone(info, annos['boxes_3d'])
                annos['ignore'] = annos.get('ignore', np.zeros(len(zone), dtype=bool)) | ~zone
            gt_annos.append(annos)
        return gt_annos

    def _region_zone(self, info, boxes):
        """(N,) bool: box centres inside the evaluated zone of a region fold (val: the val region, train: outside excluded)"""
        boxes = np.asarray(boxes)
        if len(boxes) == 0:
            return np.zeros(0, dtype=bool)
        w = region_utils.to_world(boxes[:, :2], info['world_pose'])
        return region_utils.zone_mask(self.regions, w, info['region']['fold'], info['region']['mode'], info['region'])

    def evaluation(self, det_annos, class_names, **kwargs):
        from .eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = self._eval_gt_annos()
        if self.regions is not None and any('region' in i for i in self.once_infos):
            # region fold: predictions outside the evaluated zone are dropped (neither TP nor FP)
            assert len(eval_det_annos) == len(self.once_infos)
            for anno, info in zip(eval_det_annos, self.once_infos):
                assert anno['frame_id'] == info['frame_id'] and anno.get('sequence_id', info['sequence_id']) == info['sequence_id']
                if 'region' in info and len(anno['boxes_3d']):
                    keep = self._region_zone(info, anno['boxes_3d'])
                    for key in ('name', 'score', 'boxes_3d'):
                        anno[key] = np.asarray(anno[key])[keep]
        if self.eval_force_6dof:
            for anno in eval_det_annos:
                boxes = np.asarray(anno['boxes_3d'])
                if boxes.shape[1] == 7:
                    anno['boxes_3d'] = np.concatenate([boxes, np.zeros((boxes.shape[0], 2), dtype=boxes.dtype)], axis=1)
            if self.logger is not None:
                self.logger.info('EVAL_FORCE_6DOF: 7-DoF predictions are padded with roll = pitch = 0 and scored with the 6-DoF IoU')
        ap_result_str, ap_dict = get_evaluation_results(
            eval_gt_annos, eval_det_annos, class_names,
            count_score_thresh=float(self.dataset_cfg.get('EVAL_COUNT_SCORE_THRESH', 0.3)))
        return ap_result_str, ap_dict


def create_kriso_infos(dataset_cfg, class_names, data_path, save_path, splits, workers=4, prefix='kriso_ulsan'):
    print('---------------Start to generate data infos---------------')
    for split in splits:
        cfg = copy.deepcopy(dataset_cfg)
        cfg.DATA_SPLIT = {'train': split, 'test': split}
        dataset = KrisoDataset(dataset_cfg=cfg, class_names=class_names, root_path=Path(data_path), training=False)
        assert dataset.sample_seq_list is not None, 'ImageSets/%s.txt not found' % split
        infos = dataset.get_infos(num_workers=workers)
        filename = Path(save_path) / ('%s_infos_%s.pkl' % (prefix, split))
        with open(filename, 'wb') as f:
            pickle.dump(infos, f)
        print('KRISO info %s file is saved to %s (%d frames)' % (split, filename, len(infos)))
    print('---------------Data preparation Done---------------')


def create_kriso_gt_database(data_path, infos_split, db_splits, prefix='kriso_ulsan', workers=8):
    """gt_sampling database: the points of every box of <prefix>_infos_<infos_split>.pkl are written once to
    gt_database/<sequence>_<frame>_<index>.bin (float32 x, y, z, intensity, relative to the box centre), and one index
    <prefix>_dbinfos_<split>.pkl per split in db_splits lists the boxes of the frames of that split (so the index of a
    fold's training split never contains a validation object)."""
    import concurrent.futures as futures
    data_path = Path(data_path)
    infos = pickle.load(open(data_path / ('%s_infos_%s.pkl' % (prefix, infos_split)), 'rb'))
    db_dir = data_path / 'gt_database'
    db_dir.mkdir(parents=True, exist_ok=True)

    def process(info):
        points = np.fromfile(str(data_path / 'data' / info['sequence_id'] / 'lidar_roof' / ('%s.bin' % info['frame_id'])), dtype=np.float32).reshape(-1, 4)
        points = points[np.isfinite(points).all(axis=1)]
        boxes = info['annos']['boxes_3d'].astype(np.float64)
        inside = box_rotation_utils.points_in_boxes_9dof(points[:, :3], boxes) if len(boxes) else np.zeros((0, len(points)), dtype=bool)
        entries = []
        for k in range(len(boxes)):
            obj = points[inside[k]].copy()
            obj[:, :3] -= boxes[k, :3].astype(np.float32)
            rel = 'gt_database/%s_%s_%s_%d.bin' % (info['sequence_id'], info['frame_id'], info['annos']['name'][k], k)
            obj.astype(np.float32).tofile(str(data_path / rel))
            entries.append({'name': info['annos']['name'][k], 'path': rel, 'sequence_id': info['sequence_id'], 'frame_id': info['frame_id'],
                            'gt_idx': k, 'box3d_lidar': boxes[k].astype(np.float32), 'num_points_in_gt': int(len(obj))})
        return entries

    with futures.ThreadPoolExecutor(workers) as executor:
        all_entries = [e for entries in executor.map(process, infos) for e in entries]
    print('gt database: %d boxes written to %s' % (len(all_entries), db_dir))
    for split in db_splits:
        frames = {(i['sequence_id'], i['frame_id']) for i in pickle.load(open(data_path / ('%s_infos_%s.pkl' % (prefix, split)), 'rb'))}
        db_infos = {}
        for e in all_entries:
            if (e['sequence_id'], e['frame_id']) in frames:
                db_infos.setdefault(e['name'], []).append(e)
        with open(data_path / ('%s_dbinfos_%s.pkl' % (prefix, split)), 'wb') as f:
            pickle.dump(db_infos, f)
        print('  %s_dbinfos_%s.pkl: %s' % (prefix, split, {k: len(v) for k, v in db_infos.items()}))


if __name__ == '__main__':
    import argparse
    import yaml
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, required=True, help='dataset config yaml')
    parser.add_argument('--splits', type=str, nargs='+', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--data_name', type=str, default='kriso_ulsan', help='folder under data/; the infos are named <data_name>_infos_<split>.pkl')
    args = parser.parse_args()

    dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    data_path = ROOT_DIR / 'data' / args.data_name
    create_kriso_infos(dataset_cfg, ['boat'], data_path, data_path, args.splits, args.workers, prefix=args.data_name)
