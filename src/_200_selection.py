import os
import hashlib
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

selections = get_config()['_200_selection']
CONST.PIPE200 = CONST.PIPE200.format("-".join(selections))
if not os.path.exists(CONST.PIPE200):
    os.makedirs(CONST.PIPE200)


def _201_drop_zero_variance(in_trn_path, in_tst_path,
                            out_trn_path=os.path.join(CONST.PIPE200, '_201_trn_{}.f'),
                            out_tst_path=os.path.join(CONST.PIPE200, '_201_tst_{}.f'), ):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(_hash)
    out_tst_path = out_tst_path.format(_hash)
    if get_config()['debug']:
        out_trn_path += '.debug'
        out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_dataset = pd.read_feather(in_trn_path)
    tst_dataset = pd.read_feather(in_tst_path)

    features = [c for c in trn_dataset.columns if c not in CONST.EX_COLS]
    print("Before drop zero variance features,", trn_dataset.shape)
    cols_std = trn_dataset[features].std()
    drop_features = cols_std[cols_std == 0].index.values
    trn_dataset = trn_dataset.drop(columns=drop_features)
    tst_dataset = tst_dataset.drop(columns=drop_features)
    print("After drop zero variance features,", trn_dataset.shape)

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _202_lgb_top_x(trn, tst, k):
    gsp = GroupShuffleSplit(n_splits=8, random_state=CONST.SEED)
    features = [c for c in trn.columns if c not in CONST.EX_COLS]
    feature_importance_df = pd.DataFrame()

    for ix, (train_index, valid_index) in enumerate(gsp.split(X=trn, groups=trn.EncodedEngine)):
        print("Fold", ix + 1)
        seed = CONST.SEED * ix

        X_train, y_train = trn.loc[train_index, features], trn.loc[
            train_index, 'RUL']
        X_valid, y_valid = trn.loc[valid_index, features], trn.loc[
            valid_index, 'RUL']

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=features)

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.005,
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
                          num_boost_round=1000,
                          early_stopping_rounds=40)

        print(eval_results['valid']['l1'][model.best_iteration - 1])
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model.feature_importance()
        fold_importance_df["fold"] = ix
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    cols = (feature_importance_df[
                ["feature", "importance"]
            ].groupby("feature").mean().sort_values(by="importance",
                                                    ascending=False)[:100].index)
    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
    plt.figure(figsize=(14, 25))
    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(CONST.IMPDIR,
                             f'_202_lgb_top_{k}_{get_config_name()}.png'))

    mean_feature_importance = feature_importance_df[
        ["feature", "importance"]
    ].groupby("feature").mean()
    mean_feature_importance = mean_feature_importance.sort_values(by="importance", ascending=False)

    # drop_cols = mean_feature_importance[mean_feature_importance == 0].dropna().index.values
    drop_cols = mean_feature_importance[k:].index.values

    print("Before drop features,", trn.shape)
    trn = trn.drop(columns=drop_cols)
    tst = tst.drop(columns=drop_cols)
    print("After drop features,", trn.shape)

    return trn, tst


def _203_lgb_top_100(in_trn_path, in_tst_path,
                     out_trn_path=os.path.join(CONST.PIPE200, '_203_trn_{}.f'),
                     out_tst_path=os.path.join(CONST.PIPE200, '_203_tst_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(_hash)
    out_tst_path = out_tst_path.format(_hash)
    if get_config()['debug']:
        out_trn_path += '.debug'
        out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_dataset = pd.read_feather(in_trn_path)
    tst_dataset = pd.read_feather(in_tst_path)

    le = preprocessing.LabelEncoder()
    trn_dataset['EncodedEngine'] = le.fit_transform(trn_dataset['Engine'])

    trn_dataset, tst_dataset = _202_lgb_top_x(trn_dataset, tst_dataset, k=100)

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _204_lgb_top_500(in_trn_path, in_tst_path,
                     out_trn_path=os.path.join(CONST.PIPE200, '_204_trn_{}.f'),
                     out_tst_path=os.path.join(CONST.PIPE200, '_204_tst_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(_hash)
    out_tst_path = out_tst_path.format(_hash)
    if get_config()['debug']:
        out_trn_path += '.debug'
    out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_dataset = pd.read_feather(in_trn_path)
    tst_dataset = pd.read_feather(in_tst_path)

    le = preprocessing.LabelEncoder()
    trn_dataset['EncodedEngine'] = le.fit_transform(trn_dataset['Engine'])

    trn_dataset, tst_dataset = _202_lgb_top_x(trn_dataset, tst_dataset, k=500)

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _205_lgb_top_200(in_trn_path, in_tst_path,
                     out_trn_path=os.path.join(CONST.PIPE200, '_205_trn_{}.f'),
                     out_tst_path=os.path.join(CONST.PIPE200, '_205_tst_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(_hash)
    out_tst_path = out_tst_path.format(_hash)
    if get_config()['debug']:
        out_trn_path += '.debug'
    out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_dataset = pd.read_feather(in_trn_path)
    tst_dataset = pd.read_feather(in_tst_path)

    le = preprocessing.LabelEncoder()
    trn_dataset['EncodedEngine'] = le.fit_transform(trn_dataset['Engine'])

    trn_dataset, tst_dataset = _202_lgb_top_x(trn_dataset, tst_dataset, k=200)

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _206_lgb_top_150(in_trn_path, in_tst_path,
                     out_trn_path=os.path.join(CONST.PIPE200, '_206_trn_{}.f'),
                     out_tst_path=os.path.join(CONST.PIPE200, '_206_tst_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(_hash)
    out_tst_path = out_tst_path.format(_hash)
    if get_config()['debug']:
        out_trn_path += '.debug'
    out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_dataset = pd.read_feather(in_trn_path)
    tst_dataset = pd.read_feather(in_tst_path)

    le = preprocessing.LabelEncoder()
    trn_dataset['EncodedEngine'] = le.fit_transform(trn_dataset['Engine'])

    trn_dataset, tst_dataset = _202_lgb_top_x(trn_dataset, tst_dataset, k=150)

    trn_dataset.to_feather(out_trn_path)
    tst_dataset.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _207_select_150_by_f_regression(in_trn_path, in_tst_path,
                                    out_trn_path=os.path.join(CONST.PIPE200, '_207_trn_{}.f'),
                                    out_tst_path=os.path.join(CONST.PIPE200, '_207_tst_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(_hash)
    out_tst_path = out_tst_path.format(_hash)

    if get_config()['debug']:
        out_trn_path += '.debug'
    out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn = pd.read_feather(in_trn_path)
    tst = pd.read_feather(in_tst_path)

    # impute
    tmp_trn = trn.fillna(trn.median())
    tmp_tst = tst.fillna(tst.median())

    features = [c for c in trn.columns if c not in CONST.EX_COLS]

    # Create and fit selector
    selector = SelectKBest(f_regression, k=150)
    selector.fit(tmp_trn[features], tmp_trn['RUL'])

    # Get columns to keep
    mask = selector.get_support(indices=True)
    cols = trn.columns[mask].tolist()
    new_trn = trn[['Engine', 'RUL'] + cols]
    new_tst = tst[['Engine'] + cols]

    # trn_dataset.to_feather(out_trn_path)
    # tst_dataset.to_feather(out_tst_path)

    return new_trn, new_tst


mapper = {
    "drop_zero_variance": _201_drop_zero_variance,
    "lgb_top_100": _203_lgb_top_100,
    "lgb_top_500": _204_lgb_top_500,
    "lgb_top_200": _205_lgb_top_200,
    "lgb_top_150": _206_lgb_top_150,
    "select_150_by_f_regression": _207_select_150_by_f_regression,
}


def _200_selection():
    trn_path, tst_path = _100_feature()
    for selection in selections:
        func_selection = mapper[selection]
        trn_path, tst_path = func_selection(trn_path, tst_path)
    return trn_path, tst_path


if __name__ == '__main__':
    # trn_path, tst_path = _100_feature()
    # trn_path, tst_path = _201_drop_zero_variance(trn_path, tst_path)
    trn, tst = _200_selection()
