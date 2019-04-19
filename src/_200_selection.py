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

CONST.PIPE200 = CONST.PIPE200.format(
    "-".join(str(e) for e in sum(get_config()['_200_selection'], []))
)
if not os.path.exists(CONST.PIPE200):
    os.makedirs(CONST.PIPE200)


def _201_drop_zero_variance(in_trn_path, in_tst_path,
                            out_trn_path=os.path.join(CONST.PIPE200, '_201_trn_{}_{}.f'),
                            out_tst_path=os.path.join(CONST.PIPE200, '_201_tst_{}_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(get_config_name(), _hash)
    out_tst_path = out_tst_path.format(get_config_name(), _hash)
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


def _202_drop_all_nan(in_trn_path, in_tst_path,
                      out_trn_path=os.path.join(CONST.PIPE200, '_202_trn_{}_{}.f'),
                      out_tst_path=os.path.join(CONST.PIPE200, '_202_tst_{}_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(get_config_name(), _hash)
    out_tst_path = out_tst_path.format(get_config_name(), _hash)
    if get_config()['debug']:
        out_trn_path += '.debug'
        out_tst_path += '.debug'
    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn = pd.read_feather(in_trn_path)
    tst = pd.read_feather(in_tst_path)

    # Delete all null columns
    trn = trn.replace([np.inf, -np.inf], np.nan)
    all_nan_columns = trn.columns[trn.isnull().all()].tolist()

    print("Before drop all nan features,", trn.shape)
    trn = trn.drop(columns=all_nan_columns)
    tst = tst.drop(columns=all_nan_columns)
    print("After drop all nan features,", trn.shape)

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _203_lgb_top_k(in_trn_path, in_tst_path, k,
                   out_trn_path=os.path.join(CONST.PIPE200, '_203_trn_{}_{}.f'),
                   out_tst_path=os.path.join(CONST.PIPE200, '_203_tst_{}_{}.f')):
    print("##############################")
    print("_200_selection._203_lgb_top_k ")
    print("####################k#########")
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]

    out_trn_path = out_trn_path.format(get_config_name(), _hash)
    out_tst_path = out_tst_path.format(get_config_name(), _hash)

    if get_config()['debug']:
        out_trn_path += '.debug'
        out_tst_path += '.debug'

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn = pd.read_feather(in_trn_path)
    tst = pd.read_feather(in_tst_path)

    le = preprocessing.LabelEncoder()
    trn['EncodedEngine'] = le.fit_transform(trn['Engine'])

    cv_id = get_cv_id(CONST.SEED)
    trn = trn.merge(cv_id, on=['Engine'], how='left')
    assert trn.cv_id.notnull().all()

    features = [c for c in trn.columns if c not in CONST.EX_COLS]
    feature_importance_df = pd.DataFrame()

    for i in trn.cv_id.unique():
        print("CV ID", i)
        seed = CONST.SEED * i

        X_train, y_train = trn.loc[trn.cv_id != i, features], trn.loc[trn.cv_id != i, 'RUL']
        X_valid, y_valid = trn.loc[trn.cv_id == i, features], trn.loc[trn.cv_id == i, 'RUL']

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=features)

        params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.01,
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
        fold_importance_df["fold"] = i
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
    drop_cols = list(mean_feature_importance[k:].index.values)

    print("Before drop lgbm selection features,", trn.shape)
    trn = trn.drop(columns=drop_cols + ['cv_id'])
    tst = tst.drop(columns=drop_cols)
    print("After drop lgb_selection features,", trn.shape)

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _205_lasso_selection(in_trn_path, in_tst_path, alpha=0.01,
                         out_trn_path=os.path.join(CONST.PIPE200, '_205_trn_{}_{}.f'),
                         out_tst_path=os.path.join(CONST.PIPE200, '_205_tst_{}_{}.f')):
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


def _206_ridge_selection(in_trn_path, in_tst_path, alpha=0.01,
                         out_trn_path=os.path.join(CONST.PIPE200, '_206_trn_{}_{}.f'),
                         out_tst_path=os.path.join(CONST.PIPE200, '_206_tst_{}_{}.f')):
    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:5]
    out_trn_path = out_trn_path.format(get_config_name(), _hash)
    out_tst_path = out_tst_path.format(get_config_name(), _hash)

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn = pd.read_feather(in_trn_path)
    tst = pd.read_feather(in_tst_path)

    trn = trn.fillna(trn.median())
    tst = tst.fillna(trn.median())

    features = [c for c in trn.columns if c not in CONST.EX_COLS]

    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import Ridge
    estimator = Ridge(alpha=alpha, normalize=True)
    featureSelection = SelectFromModel(estimator, threshold=1e-5)
    featureSelection.fit(trn[features], trn['RUL'])
    drop_cols = trn[features].columns[~featureSelection.get_support(indices=False)].tolist()

    print("Before drop selection by ridge regression,", trn.shape)
    trn = trn.drop(columns=drop_cols)
    tst = tst.drop(columns=drop_cols)
    print("After drop selection by ridge regression,", trn.shape)

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


mapper = {
    "drop_zero_variance": _201_drop_zero_variance,
    "drop_all_nan": _202_drop_all_nan,
    "lgb_top_k": _203_lgb_top_k,
    "lasso": _205_lasso_selection,
    "ridge": _206_ridge_selection,
}


def _200_selection():
    trn_path, tst_path = _100_feature()
    for selection in get_config()['_200_selection']:
        func_selection = mapper[selection[0]]
        if len(selection) == 1:
            trn_path, tst_path = func_selection(trn_path, tst_path)
        else:
            trn_path, tst_path = func_selection(trn_path, tst_path, *selection[1:])
    return trn_path, tst_path


if __name__ == '__main__':
    trn_path, tst_path = _200_selection()
    # trn_path, tst_path = _201_drop_zero_variance(trn_path, tst_path)
