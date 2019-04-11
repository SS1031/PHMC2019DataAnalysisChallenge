import os
import json
import optuna
import hashlib
import pandas as pd

import CONST
from utils import get_config_name
from lgb_cv import lgb_n_fold_cv_random_gs
from lgb_cv import lgb_cv_id_fold
from _200_selection import _200_selection

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


def _300_optimize(out_path=os.path.join(CONST.PIPE300, 'optimized_params_{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            params['num_leaves'] = trial.suggest_int('num_leaves', 10, 150)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 20, 1000)
            params['max_bin'] = trial.suggest_int('max_bin', 32, 512)
            params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.7, 1.0)
            params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
            params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 3)
            params['lambda_l1'] = trial.suggest_uniform('lambda_l1', .0, .1)
            params['lambda_l2'] = trial.suggest_uniform('lambda_l2', .0, .1)
            params['verbose'] = -1

            return lgb_n_fold_cv_random_gs(trn, params, folds=8)

    trn_path, tst_path = _200_selection()
    _hash = hashlib.md5((trn_path + tst_path).encode('utf-8')).hexdigest()[:5]
    out_path = out_path.format(get_config_name(), _hash)

    if os.path.exists(out_path):
        return out_path, trn_path, tst_path

    trn = pd.read_feather(trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    params['num_leaves'] = study.best_params['num_leaves']
    params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    params['max_bin'] = study.best_params['max_bin']
    params['feature_fraction'] = study.best_params['feature_fraction']
    params['bagging_fraction'] = study.best_params['bagging_fraction']
    params['lambda_l1'] = study.best_params['lambda_l1']
    params['learning_rate'] = 0.005

    with open(out_path, 'w') as fp:
        json.dump(params, fp)

    return out_path, trn_path, tst_path


def _301_optimize_cv_id(out_path=os.path.join(CONST.PIPE300, 'cv_id_optimized_params_{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            params['num_leaves'] = trial.suggest_int('num_leaves', 10, 150)
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 20, 1000)
            params['max_bin'] = trial.suggest_int('max_bin', 32, 512)
            params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.7, 1.0)
            params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
            params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 3)
            params['lambda_l1'] = trial.suggest_uniform('lambda_l1', .0, .1)
            params['lambda_l2'] = trial.suggest_uniform('lambda_l2', .0, .1)
            params['verbose'] = -1

            return lgb_cv_id_fold(trn, params)

    trn_path, tst_path = _200_selection()
    _hash = hashlib.md5((trn_path + tst_path).encode('utf-8')).hexdigest()[:5]
    out_path = out_path.format(get_config_name(), _hash)

    if os.path.exists(out_path):
        return out_path, trn_path, tst_path

    trn = pd.read_feather(trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    params['num_leaves'] = study.best_params['num_leaves']
    params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    params['max_bin'] = study.best_params['max_bin']
    params['feature_fraction'] = study.best_params['feature_fraction']
    params['bagging_fraction'] = study.best_params['bagging_fraction']
    params['lambda_l1'] = study.best_params['lambda_l1']
    params['learning_rate'] = 0.01
    params['verbose'] = -1

    with open(out_path, 'w') as fp:
        json.dump(params, fp)

    return out_path, trn_path, tst_path


if __name__ == '__main__':
    param_path, trn_path, tst_path = _300_optimize()
