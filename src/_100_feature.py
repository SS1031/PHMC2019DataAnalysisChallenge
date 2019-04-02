import os
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from tsfresh import extract_features
from tqdm import tqdm
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
import tsfresh
from sklearn import preprocessing

import CONST
from _000_preprocess import _000_preprocess
from utils import get_config

mapper = {
    "comprehensive": ComprehensiveFCParameters,
    "efficient": EfficientFCParameters,
    "minimal": MinimalFCParameters,
}

fc_parameter = mapper[get_config()['_100_feature']]()

CONST.PIPE100 = CONST.PIPE100.format(fc_parameter.__class__.__name__)
if not os.path.exists(CONST.PIPE100):
    os.makedirs(CONST.PIPE100)


def create_dataset(trn_base_path, tst_base_path,
                   trn_output_path=os.path.join(CONST.PIPE100, 'trn.f'),
                   tst_output_path=os.path.join(CONST.PIPE100, 'tst.f')):
    if get_config()['debug']:
        trn_output_path += '.debug'
        tst_output_path += '.debug'

    if os.path.exists(trn_output_path) and os.path.exists(tst_output_path):
        return trn_output_path, tst_output_path

    trn_base = pd.read_csv(trn_base_path)
    tst_base = pd.read_csv(tst_base_path)

    if get_config()['debug']:
        t_no_list = list(range(10, 350, 100))
    else:
        # 2019-04-02 テストデータのFlightNo maxを元としてデータを作成
        t_no_list = tst_base.groupby('Engine').FlightNo.max().values.tolist()
        t_no_list += list(range(10, 350, 5))
        t_no_list = list(set(t_no_list))

        # t_no_list = list(range(10, 350, 5))

    # 訓練データ作成
    trn_dataset = pd.DataFrame()
    for t_no in t_no_list:
        print('Flight NO', t_no)
        t_engine = trn_base.groupby('Engine').FlightNo.max().index[
            trn_base.groupby('Engine').FlightNo.max() >= t_no
            ].values
        tmp = trn_base[(trn_base.FlightNo <= t_no) & trn_base.Engine.isin(t_engine)]
        extracted_features = extract_features(tmp,
                                              column_id="Engine",
                                              column_sort="FlightNo",
                                              default_fc_parameters=fc_parameter)
        extracted_features['CurrentFlightNo'] = t_no

        tmp_dataset = pd.concat([
            extracted_features,
            trn_base[trn_base.Engine.isin(t_engine)].groupby(['Engine']).FlightNo.max().to_frame(
                'RUL') - t_no
        ], axis=1).reset_index().rename(columns={'index': 'Engine'})

        trn_dataset = pd.concat([trn_dataset, tmp_dataset], axis=0).reset_index(drop=True)

    # テストデータ作成
    tst_dataset = extract_features(tst_base, column_id="Engine", column_sort="FlightNo",
                                   default_fc_parameters=fc_parameter)
    tst_dataset = pd.concat([
        tst_dataset,
        tst_base.groupby('Engine').FlightNo.max().to_frame('CurrentFlightNo')],
        axis=1
    )
    # TTA的な加重平均のためのweight, weight = CurrentFlightNo / MaxFlightNo
    tst_dataset['Weight'] = 1
    # TTA用のDiffFlightNo
    tst_dataset['DiffFlightNo'] = (
            tst_base.groupby('Engine').FlightNo.max() - tst_dataset['CurrentFlightNo']
    )

    for t_no in t_no_list:
        print('Flight NO', t_no)
        t_engine = tst_base.groupby('Engine').size().index[
            tst_base.groupby('Engine').FlightNo.max() >= t_no].values
        tmp = tst_base[(tst_base.FlightNo <= t_no) & tst_base.Engine.isin(t_engine)]
        extracted_features = extract_features(tmp,
                                              column_id="Engine",
                                              column_sort="FlightNo",
                                              default_fc_parameters=fc_parameter)

        extracted_features['CurrentFlightNo'] = t_no
        # 加重平均計算のためのWeight
        extracted_features['Weight'] = t_no / tst_base.groupby('Engine').FlightNo.max()
        # TTA用のDiffFlightNo
        extracted_features['DiffFlightNo'] = (
                tst_base.groupby('Engine').FlightNo.max() - t_no
        )
        tst_dataset = pd.concat([tst_dataset, extracted_features], axis=0)

    tst_dataset = tst_dataset.reset_index().rename(columns={'index': 'Engine'})

    # TODO 2019-03-26 : 加重平均のためのWeightを導入したい
    print("Train dataset size =", trn_dataset.shape)
    print("Test dataset size =", tst_dataset.shape)
    assert (set([c for c in trn_dataset.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst_dataset.columns if c not in CONST.EX_COLS]))

    trn_dataset.to_feather(trn_output_path)
    tst_dataset.to_feather(tst_output_path)

    return trn_output_path, tst_output_path


def _100_feature():
    trn_base_path, tst_base_path = _000_preprocess()
    trn_dataset_path, tst_dataset_path = create_dataset(trn_base_path, tst_base_path)
    return trn_dataset_path, tst_dataset_path


if __name__ == '__main__':
    # trn_dataset_path, tst_dataset_path = _100_feature()
    trn_base_path, tst_base_path = _000_preprocess()
    trn_base = pd.read_csv(trn_base_path)
    tst_base = pd.read_csv(tst_base_path)

