import os
import hashlib
import pandas as pd

import CONST
from utils import get_config_name

from _100_feature import _100_feature

if not os.path.exists(CONST.PIPE200):
    os.makedirs(CONST.PIPE200)

import warnings

warnings.filterwarnings('ignore')


def _201_lasso_selection(in_trn_path, in_tst_path, seed=CONST.SEED, alpha=0.01,
                         out_trn_path=os.path.join(CONST.PIPE200, '_201_trn_seed{}_{}.f'),
                         out_tst_path=os.path.join(CONST.PIPE200, '_201_tst_seed{}_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_trn_path = out_trn_path.format(seed, _hash)
    out_tst_path = out_tst_path.format(seed, _hash)

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        print("Cache file exist")
        print(f"    {out_trn_path}")
        print(f"    {out_tst_path}")
        return out_trn_path, out_tst_path

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Lasso

    trn = pd.read_feather(in_trn_path)
    tst = pd.read_feather(in_tst_path)
    trn = trn.fillna(trn.median())

    features = [c for c in trn.columns if c not in CONST.EX_COLS]

    estimator = Lasso(alpha=alpha, normalize=True)
    featureSelection = SelectFromModel(estimator)
    featureSelection.fit(trn[features], trn['RUL'])
    drop_cols = trn[features].columns[~featureSelection.get_support(indices=False)].tolist()

    print("Before drop selection by lasso regression,", trn.shape)
    trn = trn.drop(columns=drop_cols)
    tst = tst.drop(columns=drop_cols)
    print("After drop selection by lasso regression,", trn.shape)

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _200_selection(seed=CONST.SEED):
    trn_path, tst_path = _100_feature(seed)
    return _201_lasso_selection(trn_path, tst_path, seed)


if __name__ == '__main__':
    trn_path, tst_path = _200_selection()
    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)
