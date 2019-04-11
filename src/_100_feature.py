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
from _000_preprocess import _001_preprocess
from utils import get_config

feature_set_mapper = {
    "comprehensive": ComprehensiveFCParameters,
    "efficient": EfficientFCParameters,
    "minimal": MinimalFCParameters,
}

fc_parameter = feature_set_mapper[get_config()['_100_feature']['set']]()
feature_func = get_config()['_100_feature']['func']

CONST.PIPE100 = CONST.PIPE100.format(fc_parameter.__class__.__name__, feature_func)

if not os.path.exists(CONST.PIPE100):
    os.makedirs(CONST.PIPE100)


def create_split_dataset(base_df, split_no_list=[], istest=False, kind_to_fc_parameters={}):
    dataset = pd.DataFrame()
    if istest:
        if bool(kind_to_fc_parameters):
            dataset = extract_features(base_df, column_id="Engine", column_sort="FlightNo",
                                       default_fc_parameters={},
                                       kind_to_fc_parameters=kind_to_fc_parameters)
        else:
            dataset = extract_features(base_df, column_id="Engine", column_sort="FlightNo",
                                       default_fc_parameters=fc_parameter)

        dataset = pd.concat([
            dataset,
            base_df.groupby('Engine').FlightNo.max().to_frame('CurrentFlightNo')],
            axis=1
        )
        dataset = dataset.reset_index().rename(columns={'index': 'Engine'})
        return dataset

    flight_max = base_df.groupby('Engine').FlightNo.max()
    for split_no in split_no_list:
        print('Flight NO', split_no)
        target_engine = flight_max.index[flight_max >= split_no].values
        tmp = base_df[(base_df.FlightNo <= split_no) & base_df.Engine.isin(target_engine)]

        if bool(kind_to_fc_parameters):
            extracted_features = extract_features(tmp,
                                                  column_id="Engine",
                                                  column_sort="FlightNo",
                                                  default_fc_parameters={},
                                                  kind_to_fc_parameters=kind_to_fc_parameters)
        else:
            extracted_features = extract_features(tmp,
                                                  column_id="Engine",
                                                  column_sort="FlightNo",
                                                  default_fc_parameters=fc_parameter)

        extracted_features['CurrentFlightNo'] = split_no
        tmp_dataset = pd.concat([
            extracted_features,
            base_df[base_df.Engine.isin(target_engine)].groupby(['Engine']).FlightNo.max().to_frame(
                'RUL') - split_no
        ], axis=1).reset_index().rename(columns={'index': 'Engine'})
        dataset = pd.concat([dataset, tmp_dataset], axis=0).reset_index(drop=True)

    return dataset


def create_split_regime_dataset(base_df, split_no_list=[], istest=False, kind_to_fc_parameters={}):
    dataset = pd.DataFrame()
    if istest:
        for regime in [1, 2, 3, 4, 5, 6]:
            print(f"Regime = {regime}")
            tmp_regime = base_df[base_df[f'Regime{regime}'] == 1].copy()
            tmp_regime = tmp_regime[[c for c in tmp_regime.columns if 'Regime' not in c]]

            if bool(kind_to_fc_parameters):
                regime_extracted_features = extract_features(tmp_regime,
                                                             column_id="Engine",
                                                             column_sort="FlightNo",
                                                             default_fc_parameters={},
                                                             kind_to_fc_parameters=kind_to_fc_parameters)
            else:
                regime_extracted_features = extract_features(tmp_regime,
                                                             column_id="Engine",
                                                             column_sort="FlightNo",
                                                             default_fc_parameters=fc_parameter)

            regime_extracted_features = regime_extracted_features.add_suffix(f'_Regime{regime}')
            dataset = pd.concat([dataset, regime_extracted_features], axis=1, sort=True)

        dataset = pd.concat([
            dataset,
            base_df.groupby('Engine').FlightNo.max().to_frame('CurrentFlightNo')
        ], axis=1, sort=True).reset_index().rename(columns={'index': 'Engine'})

        return dataset

    flight_max = base_df.groupby('Engine').FlightNo.max()
    for split_no in split_no_list:
        print('Flight NO', split_no)
        target_engine = flight_max.index[flight_max >= split_no].values
        tmp = base_df[(base_df.FlightNo <= split_no) & base_df.Engine.isin(target_engine)]
        regime_dataset = pd.DataFrame()
        for regime in [1, 2, 3, 4, 5, 6]:
            print(f"Regime = {regime}")
            tmp_regime = tmp[tmp[f'Regime{regime}'] == 1]
            tmp_regime = tmp_regime[[c for c in tmp_regime.columns if 'Regime' not in c]]

            if bool(kind_to_fc_parameters):
                regime_extracted_features = extract_features(tmp_regime,
                                                             column_id="Engine",
                                                             column_sort="FlightNo",
                                                             default_fc_parameters={},
                                                             kind_to_fc_parameters=kind_to_fc_parameters)
            else:
                regime_extracted_features = extract_features(tmp_regime,
                                                             column_id="Engine",
                                                             column_sort="FlightNo",
                                                             default_fc_parameters=fc_parameter)

            regime_extracted_features = regime_extracted_features.add_suffix(f'_Regime{regime}')
            regime_dataset = pd.concat([regime_dataset, regime_extracted_features], axis=1,
                                       sort=True)

        regime_dataset = pd.concat([
            regime_dataset,
            base_df[base_df.Engine.isin(target_engine)].groupby(
                ['Engine']
            ).FlightNo.max().to_frame('RUL') - split_no
        ], axis=1, sort=True).reset_index().rename(columns={'index': 'Engine'})
        regime_dataset['CurrentFlightNo'] = split_no
        dataset = pd.concat([dataset, regime_dataset], axis=0).reset_index(drop=True)

    return dataset


def _101_simple_split(in_trn_path, in_tst_path,
                      out_trn_path=os.path.join(CONST.PIPE100, 'trn.f'),
                      out_tst_path=os.path.join(CONST.PIPE100, 'tst.f')):
    if get_config()['debug']:
        out_trn_path += '.debug'
        out_tst_path += '.debug'

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_base = pd.read_csv(in_trn_path)
    tst_base = pd.read_csv(in_tst_path)

    if get_config()['debug']:
        split_no_list = list(range(20, 350, 150))
    else:
        # 2019-04-02 テストデータのFlightNo maxを元としてデータを作成
        split_no_list = tst_base.groupby('Engine').FlightNo.max().values.tolist()
        split_no_list += list(range(20, 350, 5))
        split_no_list = list(set(split_no_list))

    # 訓練データ作成
    trn = create_split_dataset(trn_base, split_no_list)
    print("Create Test Dataset")
    tst = create_split_dataset(tst_base, istest=True)

    print("Train dataset size =", trn.shape)
    print("Test dataset size =", tst.shape)
    assert (set([c for c in trn.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst.columns if c not in CONST.EX_COLS]))

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _102_regime_split(in_trn_path, in_tst_path,
                      out_trn_path=os.path.join(CONST.PIPE100, 'trn.f'),
                      out_tst_path=os.path.join(CONST.PIPE100, 'tst.f')):
    if get_config()['debug']:
        out_trn_path += '.debug'
        out_tst_path += '.debug'

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_base = pd.read_csv(in_trn_path)
    tst_base = pd.read_csv(in_tst_path)

    if get_config()['debug']:
        split_no_list = list(range(20, 350, 150))
    else:
        # 2019-04-02 テストデータのFlightNo maxを元としてデータを作成
        split_no_list = tst_base.groupby('Engine').FlightNo.max().values.tolist()
        split_no_list += list(range(20, 350, 5))
        split_no_list = list(set(split_no_list))

    # 訓練データ作成
    trn = create_split_regime_dataset(trn_base, split_no_list)
    print("Create Test Dataset")
    tst = create_split_regime_dataset(tst_base, istest=True)

    print("Train dataset size =", trn.shape)
    print("Test dataset size =", tst.shape)
    assert (set([c for c in trn.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst.columns if c not in CONST.EX_COLS]))

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _103_lgbm_selected_regime_split(in_trn_path, in_tst_path,
                                    out_trn_path=os.path.join(CONST.PIPE100, 'trn.f'),
                                    out_tst_path=os.path.join(CONST.PIPE100, 'tst.f')):
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    K = 1500
    out_feature_setting_path = os.path.join(CONST.PIPE100, f'feature_setting_top{K}.pkl')

    trn_base = pd.read_csv(in_trn_path)
    tst_base = pd.read_csv(in_tst_path)

    if os.path.exists(out_feature_setting_path):
        with open(out_feature_setting_path, 'rb') as fp:
            feature_setting = pickle.load(fp)
    else:
        ###
        # 最初にデータ選択用データセット作成する
        ###
        selection_out_trn_path = os.path.join(CONST.PIPE100, 'for_lgbm_selection_trn.f')
        selection_out_tst_path = os.path.join(CONST.PIPE100, 'for_lgbm_selection_tst.f')
        split_no_list = list(range(20, 350, 150))

        # 訓練データ作成
        print("Create Train Dataset")
        trn = create_split_regime_dataset(trn_base, split_no_list)
        print("Create Test Dataset")
        tst = create_split_regime_dataset(tst_base, istest=True)

        print("Train dataset size =", trn.shape)
        print("Test dataset size =", tst.shape)
        assert (set([c for c in trn.columns if c not in CONST.EX_COLS]) ==
                set([c for c in tst.columns if c not in CONST.EX_COLS]))

        trn.to_feather(selection_out_trn_path)
        tst.to_feather(selection_out_tst_path)

        ###
        # データ選択
        ###
        from _200_selection import _203_lgb_top_k
        selected_trn_path, selected_tst_path = _203_lgb_top_k(
            selection_out_trn_path, selection_out_tst_path, K,
            out_trn_path=os.path.join(CONST.PIPE100, 'selected_trn.f'),
            out_tst_path=os.path.join(CONST.PIPE100, 'selected_tst.f')
        )
        selected_trn = pd.read_feather(selected_trn_path)

        ###
        # 選択された特徴量で本番特徴量作成
        ###
        print("Selected Train dataset size =", selected_trn.shape)
        feat_list = [re.sub(r'_Regime[1-9]', '', col) for col in selected_trn.columns if
                     col not in CONST.EX_COLS + ['CurrentFlightNo']]
        print("Calculation feature num =", len(list(set(feat_list))))
        feature_setting = from_columns(list(set(feat_list)))

        with open(out_feature_setting_path, 'wb') as f:
            pickle.dump(feature_setting, f)

    # split_no_list = list(range(20, 350, 30))
    # 2019-04-02 テストデータのFlightNo maxを元としてデータを作成
    split_no_list = list(set(tst_base.groupby('Engine').FlightNo.max().values.tolist()))

    trn = create_split_regime_dataset(trn_base, split_no_list,
                                      kind_to_fc_parameters=feature_setting)
    tst = create_split_regime_dataset(tst_base, istest=True,
                                      kind_to_fc_parameters=feature_setting)

    print("Train dataset size =", trn.shape)
    print("Test dataset size =", tst.shape)
    assert (set([c for c in trn.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst.columns if c not in CONST.EX_COLS]))

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


func_mapper = {
    "SimpleSplit": _101_simple_split,
    "RegimeSplit": _102_regime_split,
    "LGBMSelectedRegimeSplit": _103_lgbm_selected_regime_split,
}


def _100_feature():
    trn_base_path, tst_base_path = _001_preprocess()
    trn_dataset_path, tst_dataset_path = func_mapper[feature_func](trn_base_path, tst_base_path)
    return trn_dataset_path, tst_dataset_path


if __name__ == '__main__':
    trn, tst = _100_feature()
    # trn_base_path, tst_base_path = _001_preprocess()
    # trn_path, tst_path = func_mapper[feature_func](trn_base_path, tst_base_path)
