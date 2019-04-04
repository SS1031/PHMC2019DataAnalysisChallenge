import os
import numpy as np
import pandas as pd

import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

import CONST
import utils


def lgb_n_fold_cv_random_gs(trn, params, tst=None, folds=8, seed=CONST.SEED, imp_plot=False):
    """
    tstが入っていたらpredictionとimportanceを返却する
    ランダムなgroup_shuffleで実装する
    """
    if tst is not None:
        preds = pd.DataFrame({'Engine': tst.Engine,
                              'Weight': tst.Weight,
                              'DiffFlightNo': tst.DiffFlightNo},
                             index=tst.index)
        feature_importance_df = pd.DataFrame()

    valid_preds = pd.DataFrame({'preds': [np.nan] * trn.shape[0], 'actual_RUL': trn.RUL})
    le = preprocessing.LabelEncoder()
    trn['EncodedEngine'] = le.fit_transform(trn['Engine'])
    features = [c for c in trn.columns if c not in CONST.EX_COLS]

    gsp = GroupShuffleSplit(n_splits=folds, random_state=seed)
    gsp.split(X=trn, groups=trn.EncodedEngine)
    for ix, (train_index, valid_index) in enumerate(gsp.split(X=trn, groups=trn.EncodedEngine)):
        print(f"Fold {ix + 1}")
        X_train, y_train = trn.loc[train_index, features], trn.loc[train_index, 'RUL']
        X_valid, y_valid = trn.loc[valid_index, features], trn.loc[valid_index, 'RUL']

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=features)

        eval_results = {}
        model = lgb.train(params,
                          d_train,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          evals_result=eval_results,
                          verbose_eval=100 * params['verbose'],
                          num_boost_round=10000,
                          early_stopping_rounds=40)

        valid_preds.loc[valid_index, 'preds'] = model.predict(X_valid)

        if tst is not None:
            preds[f'fold{ix + 1}'] = model.predict(tst[features])
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = features
            fold_importance_df["importance"] = model.feature_importance()
            fold_importance_df["fold"] = ix
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    if imp_plot:
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
        plt.savefig(os.path.join(CONST.IMPDIR, f'imp_{utils.get_config_name()}.png'))

    valid_preds.dropna(inplace=True)
    if tst is None:
        print("CV MAE Score :", mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds))
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds)
    else:
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds), preds


def lgb_cv_id_fold(trn, params, tst=None, seed=CONST.SEED, imp_plot=False):
    """
    tstが入っていたらpredictionとimportanceを返却する
    事前に作成したcv_idでcvを切っていく
    """
    if tst is not None:
        preds = tst[['Engine']].copy()
        feature_importance_df = pd.DataFrame()

    cv_id = utils.get_cv_id(seed)
    trn = trn.merge(cv_id, on=['Engine'], how='left')
    assert trn.cv_id.notnull().all()

    valid_preds = pd.DataFrame({'preds': [np.nan] * trn.shape[0], 'actual_RUL': trn.RUL})
    le = preprocessing.LabelEncoder()
    trn['EncodedEngine'] = le.fit_transform(trn['Engine'])
    features = [c for c in trn.columns if c not in CONST.EX_COLS]

    for i in list(range(1, 9)):
        print(f"CV ID = {i}")
        X_train, y_train = trn.loc[trn.cv_id != i, features], trn.loc[trn.cv_id != i, 'RUL']
        X_valid, y_valid = trn.loc[trn.cv_id == i, features], trn.loc[trn.cv_id == i, 'RUL']

        d_train = lgb.Dataset(X_train, label=y_train, feature_name=features)
        d_valid = lgb.Dataset(X_valid, label=y_valid, feature_name=features)

        eval_results = {}
        model = lgb.train(params,
                          d_train,
                          valid_sets=[d_train, d_valid],
                          valid_names=['train', 'valid'],
                          evals_result=eval_results,
                          verbose_eval=100 * params['verbose'],
                          num_boost_round=10000,
                          early_stopping_rounds=40)

        valid_preds.loc[trn.cv_id == i, 'preds'] = model.predict(X_valid)

        if tst is not None:
            preds[f'fold{i + 1}'] = model.predict(tst[features])
            fold_importance_df = pd.DataFrame()
            fold_importance_df["feature"] = features
            fold_importance_df["importance"] = model.feature_importance()
            fold_importance_df["fold"] = i
            feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    if imp_plot:
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
        plt.savefig(os.path.join(CONST.IMPDIR, f'imp_{utils.get_config_name()}.png'))

    valid_preds.dropna(inplace=True)
    if tst is None:
        print("CV MAE Score :", mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds))
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds)
    else:
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds), preds
