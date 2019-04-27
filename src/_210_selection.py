import os
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_selection import SelectKBest, f_regression

import CONST
from _100_feature import _100_feature
from utils import get_config
from utils import get_config_name
from utils import get_cv_id

from _110_feature import _111_offset_feature

if not os.path.exists(CONST.PIPE210):
    os.makedirs(CONST.PIPE210)


def _211_lasso_selection(alpha=0.01,
                         out_trn_path=os.path.join(CONST.PIPE210, '_211_trn_{}_{}.f'),
                         out_tst_path=os.path.join(CONST.PIPE210, '_211_tst_{}_{}.f')):
    in_trn_path, in_tst_path = _111_offset_feature()
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]

    out_trn_path = out_trn_path.format(get_config_name(), _hash)
    out_tst_path = out_tst_path.format(get_config_name(), _hash)

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
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


if __name__ == '__main__':
    trn_path, tst_path = _211_lasso_selection()
