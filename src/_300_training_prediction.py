import os
import pandas as pd
import numpy as np

import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit

import CONST
import utils
from _200_selection import _200_selection


def n_fold_cv(trn, params, tst=None, folds=8, seed=42):
    """
    tstが入っていたらpredictionとimportanceを返却する
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

    if tst is not None:
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

    if tst is None:
        valid_preds.dropna(inplace=True)
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds)
    else:
        return preds


if __name__ == '__main__':
    trn_dataset_path, tst_dataset_path = _200_selection()

    trn_dataset = pd.read_feather(trn_dataset_path)
    tst_dataset = pd.read_feather(tst_dataset_path)

    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.01,
        'num_leaves': 54,
        'min_data_in_leaf': 12,
        'max_bin': 369,
        'bagging_fraction': 0.9827855407461733,
        'lambda_l1': 0.06653735048996762,
        "bagging_seed": CONST.SEED,
        "feature_fraction_seed": CONST.SEED,
        "seed": CONST.SEED,
        "verbose": 1,
    }


    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            params['num_leaves'] = trial.suggest_int('num_leaves', 10, 150)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 100)
            params['max_bin'] = trial.suggest_int('max_bin', 32, 512)
            params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.7, 1.0)
            params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
            params['lambda_l1'] = trial.suggest_uniform('lambda_l1', .0, .1)
            params['lambda_l2'] = trial.suggest_uniform('lambda_l2', .0, .1)
            params['verbose'] = -1

            return n_fold_cv(trn, params, folds=6)


    objective = Objective(trn_dataset)
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    params['num_leaves'] = study.best_params['num_leaves']
    params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    params['max_bin'] = study.best_params['max_bin']
    params['feature_fraction'] = study.best_params['feature_fraction']
    params['bagging_fraction'] = study.best_params['bagging_fraction']
    params['lambda_l1'] = study.best_params['lambda_l1']
    params['learning_rate'] = 0.005
    print(params)

    preds = n_fold_cv(trn_dataset, params, tst_dataset)

    # TODO 2019-03-27: とりあえずweight=1だけで提出する
    weightone_preds = preds[preds.Weight == 1]

    sbmt = weightone_preds.groupby('Engine')[
        [c for c in preds.columns if 'fold' in c]
    ].mean().mean(axis=1).to_frame('Predicted RUL').reset_index()
    assert_engine = np.array(['Test' + str(i).zfill(3) for i in range(1, 101)]).astype(object)

    assert (sbmt['Engine'].values == assert_engine).all()
    sbmt[['Predicted RUL']].to_csv(os.path.join(CONST.OUTDIR, 'sbmt_{}.csv'.format(utils.get_config_name())),
                                   index=False)
