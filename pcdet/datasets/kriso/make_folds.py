"""
5-fold cross-validation split of the real KRISO data (data/kriso_ulsan).

Real sequences: data/ulsan_0609 (one recording, 1,077 frames) and data/ulsan_1105 (three bags, 1,587 frames;
data/ulsan_1105/ulsan_1105_mapping.csv gives the bag of every frame). Frame ids are in time order inside a recording.

Every recording (ulsan 0609, 1105 clockwise, 1105 counterclockwise, 1105 jangsaeng) is cut into K contiguous blocks;
fold k = block k of every recording, so each fold holds 1/K of every recording. Training of fold k drops the PURGE
frames next to each boundary of the fold-k blocks (consecutive frames are ~2.4 s apart and nearly identical; they would
leak into the validation fold otherwise).

Writes ImageSets/folds.csv and kriso_ulsan_infos_fold{k}_{train,val}.pkl from kriso_ulsan_infos_real.pkl, and
kriso_ulsan_infos_fold{k}_train_eval.pkl: as many training frames as the validation fold, evenly spaced over every
recording's training frames in time order (train AP in eval mode, no augmentation, same cost as the val evaluation)
(built first when missing), then the gt_sampling database (gt_database/) with one index per fold training split
(kriso_ulsan_dbinfos_fold{k}_train.pkl) and one for all real data (kriso_ulsan_dbinfos_real.pkl).

python -m pcdet.datasets.kriso.make_folds
"""
import csv
import pickle
from pathlib import Path

import numpy as np
import yaml
from easydict import EasyDict

from .kriso_dataset import create_kriso_gt_database, create_kriso_infos

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data' / 'kriso_ulsan'
CFG = ROOT / 'tools' / 'cfgs' / 'dataset_configs' / 'kriso_dataset.yaml'
K = 5
PURGE = 2


def recordings(infos):
    """recording name -> list of (sequence_id, frame_id) in time order"""
    rec = {}
    bag_of = {r['frame_id']: r['bag'].split('_', 2)[2] for r in csv.DictReader(open(DATA / 'data' / 'ulsan_1105' / 'ulsan_1105_mapping.csv'))}
    for info in sorted(infos, key=lambda i: (i['sequence_id'], int(i['frame_id']))):
        key = (info['sequence_id'], info['frame_id'])
        name = 'ulsan_0609' if info['sequence_id'] == 'ulsan_0609' else '1105_' + bag_of[info['frame_id']]
        rec.setdefault(name, []).append(key)
    return rec


def train_eval_subset(train, n):
    """n frames evenly spaced over the training frames (sorted by recording and time), deterministic"""
    order = sorted(train, key=lambda i: (i['sequence_id'], int(i['frame_id'])))
    idx = np.unique(np.round(np.linspace(0, len(order) - 1, n)).astype(int))
    return [order[i] for i in idx]


def main():
    if not (DATA / 'kriso_ulsan_infos_real.pkl').exists():
        create_kriso_infos(EasyDict(yaml.safe_load(open(CFG))), ['boat'], DATA, DATA, ['real'], workers=8, prefix='kriso_ulsan')
    infos = pickle.load(open(DATA / 'kriso_ulsan_infos_real.pkl', 'rb'))

    fold_of, position, purged = {}, {}, {k: set() for k in range(K)}
    for name, items in recordings(infos).items():
        for k, idx in enumerate(np.array_split(np.arange(len(items)), K)):
            for i in idx:
                fold_of[items[i]] = k
                position[items[i]] = (name, int(i))
            for edge, step in ((idx[0] - 1, -1), (idx[-1] + 1, 1)):
                for d in range(PURGE):
                    j = edge + step * d
                    if 0 <= j < len(items):
                        purged[k].add(items[j])

    with open(DATA / 'ImageSets' / 'folds.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['sequence', 'frame_id', 'recording', 'position_in_recording', 'fold'] + ['purged_from_fold%d_train' % k for k in range(K)])
        for key in sorted(fold_of):
            name, pos = position[key]
            w.writerow([key[0], key[1], name, pos, fold_of[key]] + [int(key in purged[k]) for k in range(K)])

    for k in range(K):
        val = [i for i in infos if fold_of[(i['sequence_id'], i['frame_id'])] == k]
        train = [i for i in infos if fold_of[(i['sequence_id'], i['frame_id'])] != k and (i['sequence_id'], i['frame_id']) not in purged[k]]
        pickle.dump(val, open(DATA / ('kriso_ulsan_infos_fold%d_val.pkl' % k), 'wb'))
        pickle.dump(train, open(DATA / ('kriso_ulsan_infos_fold%d_train.pkl' % k), 'wb'))
        pickle.dump(train_eval_subset(train, len(val)), open(DATA / ('kriso_ulsan_infos_fold%d_train_eval.pkl' % k), 'wb'))
        print('fold %d: val %d frames, train %d frames (%d purged)' % (k, len(val), len(train), len(infos) - len(val) - len(train)))

    create_kriso_gt_database(DATA, 'real', ['fold%d_train' % k for k in range(K)] + ['real'], prefix='kriso_ulsan')


if __name__ == '__main__':
    main()
