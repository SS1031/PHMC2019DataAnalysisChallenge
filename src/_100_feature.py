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
from utils import get_config
from _000_preprocess import _001_preprocess, _002_preprocess, _003_preprocess

from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_extraction import ComprehensiveFCParameters

import warnings

warnings.filterwarnings('ignore')

feature_set_mapper = {
    "comprehensive": ComprehensiveFCParameters,
    "minimal": MinimalFCParameters,
}
fc_parameter = feature_set_mapper[get_config()['_100_feature']['set']]()

if not os.path.exists(CONST.PIPE100):
    os.makedirs(CONST.PIPE100)


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
        r = random.randint(20, instances - 5)
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
        ], axis=0).reset_index(drop=True)

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
    for r in [1, 2, 3, 4, 5, 6]:
        print(f"Regime {r}")
        tmp = data[data.FlightRegime == r][feat_cols].reset_index(drop=True).copy()
        tmp_gb = tmp.groupby('Engine')
        remove_engines = tmp_gb.size()[tmp_gb.size() <= 1].index.values
        print("Remove Engines", remove_engines)
        _feat = _extract_features(tmp[~tmp.Engine.isin(remove_engines)], {})
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
        feat = impute(feat)

    feat.index.name = 'Engine'
    feat.reset_index(inplace=True)
    feat = feat.merge(ct, on='Engine', how='left')

    return feat


def _101_regime_feature(seed=CONST.SEED):
    out_trn_path = os.path.join(CONST.PIPE100,
                                f'_110_trn_seed{seed}_{fc_parameter.__class__.__name__}.f')
    out_tst_path = os.path.join(CONST.PIPE100,
                                f'_110_tst_seed{seed}_{fc_parameter.__class__.__name__}.f')

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        print("Cache file exist")
        print(f"    {out_trn_path}")
        print(f"    {out_tst_path}")
        return out_trn_path, out_tst_path

    trn_path, tst_path = _003_preprocess()

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


def _102_offset_feature(seed=CONST.SEED):
    out_trn_path = os.path.join(CONST.PIPE100,
                                f'_111_trn_seed{seed}_{fc_parameter.__class__.__name__}.f')
    out_tst_path = os.path.join(CONST.PIPE100,
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


feature_func_mapper = {
    "regime": _101_regime_feature,
    "offset": _102_offset_feature,
}


def _100_feature(seed=CONST.SEED):
    print("Random Seed =", seed)
    trn_path, tst_path = feature_func_mapper[get_config()['_100_feature']['func']](seed)
    return trn_path, tst_path


if __name__ == '__main__':
    trn, tst = _100_feature()
    trn = pd.read_feather(trn)
    tst = pd.read_feather(tst)
