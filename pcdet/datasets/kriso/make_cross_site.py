"""
Cross-site split of the real KRISO data (data/kriso_ulsan): train on one site, evaluate on another.

    train  cross_site_train       Bangeojin port: ulsan_0609 + ulsan_1105 clockwise + counterclockwise (all frames)
    val    cross_site_val         jangsaeng (ulsan_1105 bag 165321_jangsaeng, a different site; all frames)
           cross_site_train_eval  as many training frames as val, evenly spaced in time (train AP, no augmentation)

Every ship longer than 40 m is in jangsaeng, so the training site has none: evaluate with
--set DATA_CONFIG.EVAL_IGNORE_LENGTH_ABOVE 40.0 (those ships become "don't care").
No purge is needed: no training frame is from the validation site.

Writes kriso_ulsan_infos_cross_site_{train,val,train_eval}.pkl (from kriso_ulsan_infos_real.pkl) and the gt_sampling
index kriso_ulsan_dbinfos_cross_site_train.pkl (entries of kriso_ulsan_dbinfos_real.pkl from training frames only,
the object files in gt_database/ are shared).

python -m pcdet.datasets.kriso.make_cross_site
"""
import csv
import pickle
from collections import Counter
from pathlib import Path

from .make_folds import train_eval_subset

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'data' / 'kriso_ulsan'
VAL_BAG = 'jangsaeng'


def site_of(info, bag_of):
    if info['sequence_id'] == 'ulsan_0609':
        return 'bangeojin'
    return 'jangsaeng' if bag_of[info['frame_id']].endswith(VAL_BAG) else 'bangeojin'


def main():
    bag_of = {r['frame_id']: r['bag'] for r in csv.DictReader(open(DATA / 'data' / 'ulsan_1105' / 'ulsan_1105_mapping.csv'))}
    infos = pickle.load(open(DATA / 'kriso_ulsan_infos_real.pkl', 'rb'))
    train = [i for i in infos if site_of(i, bag_of) == 'bangeojin']
    val = [i for i in infos if site_of(i, bag_of) == 'jangsaeng']
    train_eval = train_eval_subset(train, len(val))
    for name, part in (('train', train), ('val', val), ('train_eval', train_eval)):
        pickle.dump(part, open(DATA / ('kriso_ulsan_infos_cross_site_%s.pkl' % name), 'wb'))
        print('cross_site_%-10s %4d frames %s' % (name, len(part), dict(Counter(
            'ulsan_0609' if i['sequence_id'] == 'ulsan_0609' else bag_of[i['frame_id']].split('_', 2)[2] for i in part))))

    frames = {(i['sequence_id'], i['frame_id']) for i in train}
    db = pickle.load(open(DATA / 'kriso_ulsan_dbinfos_real.pkl', 'rb'))
    db_train = {cls: [e for e in entries if (e['sequence_id'], e['frame_id']) in frames] for cls, entries in db.items()}
    pickle.dump(db_train, open(DATA / 'kriso_ulsan_dbinfos_cross_site_train.pkl', 'wb'))
    print('gt_sampling index cross_site_train: %s of %s objects' % (
        {k: len(v) for k, v in db_train.items()}, {k: len(v) for k, v in db.items()}))


if __name__ == '__main__':
    main()
