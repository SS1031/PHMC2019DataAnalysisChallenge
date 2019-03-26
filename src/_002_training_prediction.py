import os
import pandas as pd
import numpy as np

import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from _001_feature import _001_feature
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import CONST
import optuna
from sklearn.metrics import mean_absolute_error


def n_fold_cv(trn, params, tst=None, folds=8, seed=777):
    """
    tstが入っていたらpredictionとimportanceを返却する
    """

    if tst is not None:
        preds = pd.DataFrame({'Engine': tst.Engine,
                              'CurrentFlightNo': tst.CurrentFlightNo},
                             index=tst.index)
        feature_importance_df = pd.DataFrame()

    valid_preds = pd.DataFrame({'preds': [np.nan] * trn.shape[0], 'actual_RUL': trn.RUL})

    le = preprocessing.LabelEncoder()
    trn['EncodedEngine'] = le.fit_transform(trn['Engine'])
    features = [c for c in trn_dataset.columns if c not in CONST.EX_COLS]

    gsp = GroupShuffleSplit(n_splits=folds, random_state=seed)
    gsp.split(X=trn, groups=trn.EncodedEngine)
    for ix, (train_index, valid_index) in enumerate(gsp.split(X=trn, groups=trn.EncodedEngine)):
        print(f"Fold {ix + 1}")
        seed = seed * ix

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
                          verbose_eval=100,
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

    if tst is not None:
        preds['Predicted RUL'] = preds[[c for c in preds.columns if 'fold' in c]].mean(axis=1)
        preds['Predicted Total Life'] = preds['Predicted RUL'] + preds['CurrentFlightNo']

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
        plt.savefig(os.path.join(CONST.IMPDIR, f'imp_002.png'))

    if tst is None:
        valid_preds.dropna(inplace=True)
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds)
    else:
        return preds[['Engine', 'Predicted RUL', 'Predicted Total Life']]


if __name__ == '__main__':
    trn_dataset_path, tst_dataset_path = _001_feature()

    trn_dataset = pd.read_feather(trn_dataset_path)
    tst_dataset = pd.read_feather(tst_dataset_path)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.01,
        'num_leaves': 54,
        "verbose": 1,
        'min_data_in_leaf': 12,
        'max_bin': 369,
        'bagging_fraction': 0.9827855407461733,
        'lambda_l1': 0.06653735048996762,
        "bagging_seed": 777,
        "feature_fraction_seed": 777,
        "seed": 777,
    }

    #
    # class Objective(object):
    #     def __init__(self, trn, tst):
    #         self.trn = trn
    #
    #     def __call__(self, trial):
    #         trn = self.trn
    #
    #         params['num_leaves'] = trial.suggest_int('num_leaves', 10, 100)
    #         params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 50)
    #         params['max_bin'] = trial.suggest_int('max_bin', 64, 512)
    #         params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
    #         params['lambda_l1'] = trial.suggest_uniform('lambda_l1', .0, .1)
    #         # params['lambda_l2'] = trial.suggest_uniform('lambda_l2', .0, .1)
    #         params['verbose'] = -1
    #
    #         return n_fold_cv(trn, params, folds=6)
    #
    #
    # objective = Objective(trn_dataset, tst_dataset)
    # study = optuna.create_study()
    # study.optimize(objective, n_trials=10)
    # params['num_leaves'] = study.best_params['num_leaves']
    # params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    # params['max_bin'] = study.best_params['max_bin']
    # params['bagging_fraction'] = study.best_params['bagging_fraction']
    # params['learning_rate'] = 0.01
    preds = n_fold_cv(trn_dataset, params, tst_dataset)
    preds[['Predicted RUL']].to_csv(os.path.join(CONST.OUTDIR, 'sbmt_24_03_2019.csv'), index=False)
