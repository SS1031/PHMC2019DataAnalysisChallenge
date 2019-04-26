import os
import json
import optuna
import hashlib
import pandas as pd

import CONST
from utils import get_config_name
from lgb_cv import lgb_n_fold_cv_random_gs
from lgb_cv import lgb_cv_id_fold
from _200_selection import _200_selection
from _110_feature import _110_regime_feature, _111_offset_feature

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "l1",
    "learning_rate": 0.01,
    "seed": CONST.SEED,
    "verbose": 1
}

trn_path, tst_path = _110_regime_feature()
_110_trn = pd.read_feather(trn_path)
_110_tst = pd.read_feather(tst_path)

trn_path, tst_path = _111_offset_feature()
_111_trn = pd.read_feather(trn_path)
_111_tst = pd.read_feather(tst_path)

trn = pd.concat([_110_trn.set_index('Engine'),
                 _111_trn.set_index('Engine')[
                     [f for f in _111_trn.columns if f not in CONST.EX_COLS + ['CutoffFlight']]]], axis=1)

tst = pd.concat([_110_tst.set_index('Engine'),
                 _111_tst.set_index('Engine')[[f for f in _111_tst.columns if f not in CONST.EX_COLS]]], axis=1)

score = lgb_cv_id_fold(trn, params)
