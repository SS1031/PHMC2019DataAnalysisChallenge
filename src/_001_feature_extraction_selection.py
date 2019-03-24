import os
import pandas as pd
import CONST
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from tsfresh import extract_features
from tqdm import tqdm
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
import tsfresh
import json

from sklearn import preprocessing

trn_base = pd.read_csv(os.path.join(CONST.INDIR, 'trn_base.csv'))
tst_base = pd.read_csv(os.path.join(CONST.INDIR, 'tst_base.csv'))

trn_regime = pd.get_dummies(trn_base['FlightRegime'])
trn_regime.columns = [f'Regime{c}' for c in trn_regime.columns]
trn_base = pd.concat([trn_base, trn_regime], axis=1)
trn_base = trn_base.drop(columns=['FlightRegime'])

tst_regime = pd.get_dummies(tst_base['FlightRegime'])
tst_regime.columns = [f'Regime{c}' for c in tst_regime.columns]
tst_base = pd.concat([tst_base, tst_regime], axis=1)
tst_base = tst_base.drop(columns=['FlightRegime'])

base_fc_parameter = EfficientFCParameters()
CONST.FEATDIR001 = CONST.FEATDIR001.format(base_fc_parameter.__class__.__name__)

if not os.path.exists(CONST.FEATDIR001):
    os.makedirs(CONST.FEATDIR001)


def extract_tsfresh_features(output_path=os.path.join(CONST.FEATDIR001, 'all_tsfeature.f')):
    if os.path.exists(output_path):
        return output_path

    dataset = pd.DataFrame()
    t_no_list = [50, 150, 250]
    for t_no in t_no_list:
        print('Flight NO', t_no)
        t_engine = trn_base.groupby('Engine').size().index[
            trn_base.groupby('Engine').FlightNo.max() >= t_no
            ].values

        tmp = trn_base[(trn_base.FlightNo <= t_no) & trn_base.Engine.isin(t_engine)]
        extracted_features = extract_features(tmp, column_id="Engine", column_sort="FlightNo",
                                              default_fc_parameters=base_fc_parameter)
        extracted_features['CurrentFlightNo'] = t_no

        tmp_dataset = pd.concat([
            extracted_features,
            trn_base[trn_base.Engine.isin(t_engine)].groupby(['Engine']).FlightNo.max().to_frame('RUL') - t_no
        ], axis=1).reset_index().rename(columns={'index': 'Engine'})

        dataset = pd.concat([dataset, tmp_dataset], axis=0).reset_index(drop=True)

    dataset.to_feather(output_path)

    return output_path


def drop_zero_stddev_features(input_path, output_path=os.path.join(CONST.FEATDIR001, 'drop_zero_std_tsfeaure.f')):
    if os.path.exists(output_path):
        return output_path

    dataset = pd.read_feather(input_path)

    print("Before drop features,", dataset.shape)
    cols_std = dataset.std()
    drop_features = cols_std[cols_std == 0].index.values
    dataset = dataset.drop(columns=drop_features)
    print("After drop features,", dataset.shape)

    dataset.to_feather(output_path)

    return output_path


def feature_selection_by_lgbm(input_path, output_path=os.path.join(CONST.FEATDIR001, 'lgb_selected_tsfeature.f')):
    if os.path.exists(output_path):
        return output_path
    seed = 777
    dataset = pd.read_feather(input_path)
    le = preprocessing.LabelEncoder()
    dataset['EncodedEngine'] = le.fit_transform(dataset['Engine'])

    gsp = GroupShuffleSplit(n_splits=8, random_state=seed)

    features = [c for c in dataset.columns if c not in CONST.EX_COLS]
    feature_importance_df = pd.DataFrame()
    for ix, (train_index, valid_index) in enumerate(gsp.split(X=dataset, groups=dataset.EncodedEngine)):
        print("Fold", ix + 1)
        seed = seed * ix
        X_train, y_train = dataset.loc[train_index, features], dataset.loc[train_index, 'RUL']
        X_valid, y_valid = dataset.loc[valid_index, features], dataset.loc[valid_index, 'RUL']

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=features)

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.01,
            # "feature_fraction": 0.9,
            # "bagging_fraction": 0.8,
            # "bagging_freq": 5,
            "verbose": 1,
            "bagging_seed": seed,
            "feature_fraction_seed": seed,
            "seed": seed,
        }
        eval_results = {}
        model = lgb.train(params,
                          d_train,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          evals_result=eval_results,
                          verbose_eval=100,
                          num_boost_round=100,
                          early_stopping_rounds=40)

        print(eval_results['valid']['l1'][model.best_iteration - 1])
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = ix
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    mean_feature_importance = feature_importance_df[
        ["feature", "importance"]
    ].groupby("feature").mean()

    drop_cols = mean_feature_importance[mean_feature_importance == 0].dropna().index.values
    print("Before drop features,", dataset.shape)
    dataset = dataset.drop(columns=drop_cols)
    print("After drop features,", dataset.shape)

    dataset.to_feather(output_path)

    return output_path


def feature_to_fc_settting_dict(input_path, output_path=os.path.join(CONST.FEATDIR001, 'lgb_selected_setting.json')):
    # if os.path.exists(output_path):
    #     return output_path

    dataset = pd.read_feather(input_path)
    features = [c for c in dataset.columns if c not in CONST.EX_COLS]
    fc_dict = tsfresh.feature_extraction.settings.from_columns(dataset[features])

    # with open(output_path, 'w') as f:
    #     json.dump(fc_dict, f)

    return fc_dict


def create_dataset(fc_setting, trn_output_path=os.path.join(CONST.FEATDIR001, 'trn_dataset.f'),
                   tst_output_path=os.path.join(CONST.FEATDIR001, 'tst_dataset.f')):
    if os.path.exists(trn_output_path) and os.path.exists(tst_output_path):
        return trn_output_path, tst_output_path

    # with open(input_path, 'r') as f:
    #     fc_setting = json.load(f)

    trn_dataset = pd.DataFrame()
    t_no_list = list(range(20, 350, 5))
    for t_no in t_no_list:
        print('Flight NO', t_no)
        t_engine = trn_base.groupby('Engine').size().index[
            trn_base.groupby('Engine').FlightNo.max() >= t_no
            ].values

        tmp = trn_base[(trn_base.FlightNo <= t_no) & trn_base.Engine.isin(t_engine)]
        extracted_features = extract_features(tmp,
                                              column_id="Engine",
                                              column_sort="FlightNo",
                                              default_fc_parameters={},  # これ空の辞書を入れないとdefault特徴量も作られる
                                              kind_to_fc_parameters=fc_setting)

        extracted_features['CurrentFlightNo'] = t_no

        tmp_dataset = pd.concat([
            extracted_features,
            trn_base[trn_base.Engine.isin(t_engine)].groupby(['Engine']).FlightNo.max().to_frame('RUL') - t_no
        ], axis=1).reset_index().rename(
            columns={'index': 'Engine'}
        )

        trn_dataset = pd.concat([trn_dataset, tmp_dataset], axis=0).reset_index(drop=True)

    tst_dataset = extract_features(tst_base, column_id="Engine", column_sort="FlightNo",
                                   default_fc_parameters={}, kind_to_fc_parameters=fc_setting)
    tst_dataset = pd.concat([
        tst_dataset,
        tst_base.groupby('Engine').FlightNo.max().to_frame('CurrentFlightNo')], axis=1).reset_index().rename(
        columns={'index': 'Engine'}
    )

    print(trn_dataset.shape)
    print(tst_dataset.shape)
    assert (set([c for c in trn_dataset.columns if c not in CONST.EX_COLS]) ==
            set([c for c in tst_dataset.columns if c not in CONST.EX_COLS]))

    trn_dataset.to_feather(trn_output_path)
    tst_dataset.to_feather(tst_output_path)

    return trn_output_path, tst_output_path


def _001_pipeline():
    dataset_path = extract_tsfresh_features()
    dataset_path = drop_zero_stddev_features(dataset_path)
    dataset_path = feature_selection_by_lgbm(dataset_path)
    fc_dict = feature_to_fc_settting_dict(dataset_path)
    trn_dataset_path, tst_dataset_path = create_dataset(fc_dict)

    return trn_dataset_path, tst_dataset_path


if __name__ == '__main__':
    trn_dataset_path, tst_dataset_path = _001_pipeline()
