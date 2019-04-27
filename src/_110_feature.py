"""一度だけのカットオフで特徴量作成
"""

import os
import re
import pandas as pd
import numpy as np
import random
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

import CONST
from _000_preprocess import _001_preprocess, _002_preprocess
from _000_preprocess import _010_preprocess

from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import ComprehensiveFCParameters

fc_parameter = MinimalFCParameters()

if not os.path.exists(CONST.PIPE110):
    os.makedirs(CONST.PIPE110)


def new_labels(data, seed):
    ct_ids = []
    ct_flights = []
    ct_labels = []

    regex = re.compile('[^0-9]')
    data = data.copy()
    gb = data.groupby(['Engine'])
    for engine, engine_no_df in gb:
        instances = engine_no_df.shape[0]
        random.seed(seed + int(regex.sub('', engine)))
        r = random.randint(5, instances - 3)
        ct_ids.append(engine_no_df.iloc[r, :]['Engine'])
        ct_flights.append(engine_no_df.iloc[r, :]['FlightNo'])
        ct_labels.append(engine_no_df.iloc[r, :]['RUL'])

    ct = pd.DataFrame({'Engine': ct_ids,
                       'CutoffFlight': ct_flights,
                       'RUL': ct_labels})
    ct = ct[['Engine', 'CutoffFlight', 'RUL']]

    return ct


def make_cutoff_flights(data, seed=CONST.SEED):
    data['RUL'] = np.nan
    for eng in data.Engine.unique():
        data.loc[(data.Engine == eng), 'RUL'] = (
                data[(data.Engine == eng)].FlightNo.max() - data.loc[(data.Engine == eng), 'FlightNo']
        )
    assert data['RUL'].notnull().all()
    data['RUL'] = data['RUL'].astype(int)

    ct = new_labels(data, seed)
    ct.index = ct['Engine']
    assert ((ct['CutoffFlight'] + ct['RUL']) == data.groupby('Engine').FlightNo.max()).all()
    ct.reset_index(inplace=True, drop=True)

    return ct


def make_cutoff_data(ct, data):
    base_data = pd.DataFrame()
    for row in ct.itertuples():
        base_data = pd.concat([
            base_data,
            data[(data.Engine == row.Engine) & (data.FlightNo <= row.CutoffFlight)].copy()
        ], axis=0)

    return base_data


def _extract_features(df, kind_to_fc_parameters={}):
    if bool(kind_to_fc_parameters):
        _feature = extract_features(df, column_id="Engine", column_sort="FlightNo",
                                    default_fc_parameters={},
                                    kind_to_fc_parameters=kind_to_fc_parameters)
    else:
        _feature = extract_features(df, column_id="Engine", column_sort="FlightNo",
                                    default_fc_parameters=fc_parameter)
    return _feature


def tsfresh_extract_cutoff_feature(data, seed, istest=False, feature_setting={}):
    if istest:
        ct = data.groupby('Engine').FlightNo.max().rename('CutoffFlight').reset_index()
    else:
        ct = make_cutoff_flights(data.copy(), seed)
        data = make_cutoff_data(ct, data)

    feat = _extract_features(data, feature_setting)
    feat = impute(feat)
    feat.index.name = 'Engine'
    feat.reset_index(inplace=True)
    feat = feat.merge(ct, on='Engine', how='left')
    feat.set_index('Engine', inplace=True)
    feat_cols = [f for f in feat.columns if f not in CONST.EX_COLS]

    if not istest:
        print("Extracted Feature Shape =", feat.shape)
        print("First Step Selection...")
        _feat = select_features(feat[feat_cols], feat['RUL'], ml_task='regression')
        print("Selected Feature Shape =", _feat.shape)
        feat = pd.concat([_feat, feat['RUL']], axis=1)

    feat.reset_index(inplace=True)
    return feat


def tsfresh_extract_cutoff_regime_feature(data, seed, istest=False):
    feat = pd.DataFrame()
    if istest:
        print("Test feature processing...")
        ct = data.groupby('Engine').FlightNo.max().rename('CutoffFlight').reset_index()
    else:
        print("Train feature processing...")
        ct = make_cutoff_flights(data.copy(), seed)
        data = make_cutoff_data(ct, data)

    feat_cols = [f for f in data.columns if f not in ['FlightRegime']]
    for r in [1, 2, 3, 5, 6]:
        print(f"Regime {r}")
        _feat = _extract_features(data[data.FlightRegime == r][feat_cols], {})
        _feat = impute(_feat)
        if not istest:
            _feat.index.name = 'Engine'
            _feat.reset_index(inplace=True)
            _feat = _feat.merge(ct[['Engine', 'RUL']], on='Engine', how='left')
            _feat.set_index('Engine', inplace=True)
            print("Extracted Feature Shape =", _feat.shape)
            print("First Step Selection...")
            _feat_cols = [f for f in _feat.columns if f not in CONST.EX_COLS]
            _feat = select_features(_feat[_feat_cols], _feat['RUL'], ml_task='regression')
            print("Selected Feature Shape =", _feat.shape)
        _feat.columns = [c + f'_Regime{r}' for c in _feat.columns]
        feat = pd.concat([feat, _feat], axis=1, sort=True)

    feat.index.name = 'Engine'
    feat.reset_index(inplace=True)
    feat = feat.merge(ct, on='Engine', how='left')

    return feat


def _110_regime_feature(seed=CONST.SEED):
    out_trn_path = os.path.join(CONST.PIPE110,
                                f'_111_trn_seed{seed}_{fc_parameter.__class__.__name__}.f')
    out_tst_path = os.path.join(CONST.PIPE110,
                                f'_111_tst_seed{seed}_{fc_parameter.__class__.__name__}.f')

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_path, tst_path = _010_preprocess()

    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)
    trn_dataset = tsfresh_extract_cutoff_regime_feature(trn, seed)
    tst_dataset = tsfresh_extract_cutoff_regime_feature(tst, seed, istest=True)

    feat_cols = [c for c in trn_dataset.columns if c not in CONST.EX_COLS]
    tst_dataset = tst_dataset[['Engine'] + feat_cols]
    assert (set([c for c in trn_dataset.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst_dataset.columns if c not in CONST.EX_COLS]))

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _111_offset_feature(seed=CONST.SEED):
    out_trn_path = os.path.join(CONST.PIPE110,
                                f'_111_trn_seed{seed}_{fc_parameter.__class__.__name__}.f')
    out_tst_path = os.path.join(CONST.PIPE110,
                                f'_111_tst_seed{seed}_{fc_parameter.__class__.__name__}.f')

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_path, tst_path = _002_preprocess()
    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    trn_dataset = tsfresh_extract_cutoff_feature(trn, seed)
    tst_dataset = tsfresh_extract_cutoff_feature(tst, seed, istest=True)
    feat_cols = [c for c in trn_dataset.columns if c not in CONST.EX_COLS]
    tst_dataset = tst_dataset[['Engine'] + feat_cols]
    assert (set([c for c in trn_dataset.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst_dataset.columns if c not in CONST.EX_COLS]))

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


if __name__ == '__main__':
    trn, tst = _111_offset_feature()
    trn = pd.read_feather(trn)
    tst = pd.read_feather(tst)
