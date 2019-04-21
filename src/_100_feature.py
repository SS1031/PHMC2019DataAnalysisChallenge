import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from tsfresh import extract_features
from tqdm import tqdm
import re
import pickle
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
import tsfresh
from sklearn import preprocessing

import CONST
from _000_preprocess import _000_mapper
# from _000_preprocess import _002_preprocess
from utils import get_config

feature_set_mapper = {
    "comprehensive": ComprehensiveFCParameters,
    "efficient": EfficientFCParameters,
    "minimal": MinimalFCParameters,
}

fc_parameter = feature_set_mapper[get_config()['_100_feature']['set']]()
FEATURE_TYPES = get_config()['_100_feature']['type']

trn_base_path, tst_base_path = _000_mapper['drop_useless']()
trn_base = pd.read_feather(trn_base_path)
tst_base = pd.read_feather(tst_base_path)

split_mapper = {
    "small": list(range(20, 360, 100)),
    "medium": list(range(20, 360, 30)),
    "based_on_test": list(set(tst_base.groupby('Engine').FlightNo.max().values.tolist())),
    "besed_on_train": list(set(
        list(trn_base.groupby('Engine').FlightNo.max().sort_values().values[0:200])
    )),
    "based_on_train_and_test": list(set(
        list(tst_base.groupby('Engine').FlightNo.max().values) +
        list(trn_base.groupby('Engine').FlightNo.max().sort_values().values[0:200])
    )),
    "large": list(set(
        list(range(20, 360, 5)) +
        list(tst_base.groupby('Engine').FlightNo.max().values) +
        list(trn_base.groupby('Engine').FlightNo.max().sort_values().values[0:200])
    )),
}

CONST.PIPE100 = CONST.PIPE100.format(
    "-".join([str(c) for c in get_config()['_100_feature']['preselection']]),
    fc_parameter.__class__.__name__,
    "-".join(FEATURE_TYPES),
    get_config()['_100_feature']['split']
)

if not os.path.exists(CONST.PIPE100):
    os.makedirs(CONST.PIPE100)


def _extract_features(df, kind_to_fc_parameters):
    if bool(kind_to_fc_parameters):
        _feature = extract_features(df, column_id="Engine", column_sort="FlightNo",
                                    default_fc_parameters={},
                                    kind_to_fc_parameters=kind_to_fc_parameters)
    else:
        _feature = extract_features(df, column_id="Engine", column_sort="FlightNo",
                                    default_fc_parameters=fc_parameter)
    return _feature


def _create_rul(df, split_list):
    """訓練データセット専用の関数, RULとCurrentFlightを作成する
    """
    RUL = pd.DataFrame()
    flight_max = df.groupby('Engine').FlightNo.max()
    for split in split_list:
        offset = 10
        target_engine = flight_max.index[(flight_max - offset) >= split].values
        rul = (df[df.Engine.isin(target_engine)].groupby(
            ['Engine']
        ).FlightNo.max().to_frame('RUL') - split).reset_index()
        rul['Engine-Split'] = rul['Engine'] + "-" + "{0:03d}".format(split)
        rul['CurrentFlightNo'] = split
        RUL = pd.concat([RUL, rul], axis=0).reset_index(drop=True)

    return RUL[['Engine', 'Engine-Split', 'RUL']]


def split_extract_feature(df, split_list, kind_to_fc_parameters):
    dataset = pd.DataFrame()
    flight_max = df.groupby('Engine').FlightNo.max()
    _split = []
    for split in split_list:
        print(f"Split = {split}")
        target_engine = flight_max.index[flight_max >= split].values
        tmp = df[(df.FlightNo <= split) & df.Engine.isin(target_engine)]
        _features = _extract_features(tmp, kind_to_fc_parameters)
        _features = _features.reset_index().rename(columns={_features.index.name: 'Engine'})
        _split = _split + ["{0:03d}".format(split)] * len(_features)
        dataset = pd.concat([dataset, _features], axis=0).reset_index(drop=True)

    dataset['Split'] = _split
    dataset['Engine-Split'] = dataset['Engine'] + "-" + dataset['Split']
    dataset.drop(columns=['Split'], inplace=True)

    return dataset


def create_dataset(df, split_list=[], istest=False, kind_to_fc_parameters={}):
    if istest:
        print("Create TEST dataset")
        dataset = pd.DataFrame()
    else:
        print("Create TRAIN dataset")
        dataset = _create_rul(df, split_list)

    if "all" in FEATURE_TYPES:
        print("Feature Type = ALL")
        if istest:
            _features = _extract_features(df, kind_to_fc_parameters)
            dataset = pd.concat([dataset, _features], axis=1, sort=True)
        else:
            _features = split_extract_feature(df, split_list, kind_to_fc_parameters)
            _features.drop(columns='Engine', inplace=True)
            dataset = dataset.merge(_features, on='Engine-Split', how='left')

    if "regime" in FEATURE_TYPES:
        print("Feature Type = Regime")
        for regime in [1, 2, 3, 4, 5, 6]:
            print(f"Regime = {regime}")
            regime_df = df[df[f'Regime{regime}'] == 1].copy()
            regime_df = regime_df[[c for c in regime_df.columns if 'Regime' not in c]]

            if istest:
                _features = _extract_features(regime_df, kind_to_fc_parameters)
                _features = _features.add_suffix(f'_Regime{regime}')
                dataset = pd.concat([dataset, _features], axis=1, sort=True)
            else:
                _features = split_extract_feature(regime_df, split_list, kind_to_fc_parameters)
                _features.drop(columns='Engine', inplace=True)
                feature_cols = [f for f in _features.columns if f != 'Engine-Split']
                rename_dict = dict(
                    zip(feature_cols, [f + f'_Regime{regime}' for f in feature_cols]))
                _features = _features.rename(columns=rename_dict)
                dataset = dataset.merge(_features, on='Engine-Split', how='left')

    if istest:
        dataset = dataset.reset_index().rename(columns={dataset.index.name: 'Engine'})
    else:
        dataset.drop(columns="Engine-Split", inplace=True)
        assert dataset.RUL.notnull().all()

    return dataset


def _100_feature(out_trn_path=os.path.join(CONST.PIPE100, 'trn.f'),
                 out_tst_path=os.path.join(CONST.PIPE100, 'tst.f')):
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_base_path, tst_base_path = _000_mapper[get_config()['_000_preprocess']]()

    trn_base = pd.read_feather(trn_base_path)
    tst_base = pd.read_feather(tst_base_path)

    from _200_selection import mapper as selection_mapper
    feature_setting = {}
    preselection_conf = get_config()['_100_feature']['preselection']

    if "" != preselection_conf:
        selected_out_trn_path = os.path.join(CONST.PIPE100, 'selected_trn.f')
        selected_out_tst_path = os.path.join(CONST.PIPE100, 'selected_tst.f')

        select_base_out_trn_path = os.path.join(CONST.PIPE100, 'selection_trn.f')
        select_base_out_tst_path = os.path.join(CONST.PIPE100, 'selection_tst.f')

        selection_split = list(range(20, 350, 100))
        trn = create_dataset(trn_base, selection_split, False, {})
        tst = create_dataset(tst_base, [], True, {})

        trn.to_feather(select_base_out_trn_path)
        tst.to_feather(select_base_out_tst_path)
        func_selection = selection_mapper[preselection_conf[0]]

        if len(preselection_conf) == 1:
            trn_path, tst_path = func_selection(select_base_out_trn_path, select_base_out_tst_path)
        else:
            trn_path, tst_path = func_selection(select_base_out_trn_path,
                                                select_base_out_tst_path,
                                                *preselection_conf[1:],
                                                out_trn_path=selected_out_trn_path,
                                                out_tst_path=selected_out_tst_path)

        selected_trn = pd.read_feather(trn_path)
        print("Selected Train dataset size =", selected_trn.shape)

        feat_list = [re.sub(r'_Regime[1-9]', '', col) for col in selected_trn.columns if
                     col not in CONST.EX_COLS + ['CurrentFlightNo']]
        print("Calculation feature num =", len(list(set(feat_list))))
        feature_setting = from_columns(list(set(feat_list)))

    split_list = split_mapper[get_config()['_100_feature']['split']]
    trn = create_dataset(trn_base, split_list, False, feature_setting)
    tst = create_dataset(tst_base, [], True, feature_setting)

    assert (set([c for c in trn.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst.columns if c not in CONST.EX_COLS]))

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


if __name__ == '__main__':
    print("test")
    trn_path, tst_path = _100_feature()
