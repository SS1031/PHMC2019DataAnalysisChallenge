import os
import json
import optuna
import hashlib
import pandas as pd

import CONST
from utils import get_config_name
from utils import get_config
from lgb_cv import lgb_cv_id_fold
from knn_cv import knn_cv_id_fold
from _200_selection import _200_selection

lgb_optimize_params = {
    "boosting_type": "gbdt",
    "objective": "regression_l2",
    "metric": "l2_root",
    "learning_rate": 0.01,
    'num_leaves': 54,
    'min_data_in_leaf': 12,
    'max_bin': 369,
    'bagging_fraction': 0.9827855407461733,
    'lambda_l1': 0.06653735048996762,
    "verbose": 1,
}


def _301_optimize_lgb(in_trn_path, in_tst_path, seed=CONST.SEED,
                      out_path=os.path.join(CONST.PIPE300, '_301_lgb_optimized_params_seed{}_{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            lgb_optimize_params['num_leaves'] = trial.suggest_int('num_leaves', 2, 128)
            lgb_optimize_params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 128)
            lgb_optimize_params['max_bin'] = trial.suggest_int('max_bin', 1, 255)
            lgb_optimize_params['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.7, 1.0)
            lgb_optimize_params['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
            lgb_optimize_params['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 3)
            lgb_optimize_params['bagging_seed'] = seed
            lgb_optimize_params['feature_fraction_seed'] = seed
            lgb_optimize_params['seed'] = seed
            # params['lambda_l1'] = trial.suggest_uniform('lambda_l1', .0, .1)
            # params['lambda_l2'] = trial.suggest_uniform('lambda_l2', .0, .1)
            lgb_optimize_params['verbose'] = -1

            return lgb_cv_id_fold(trn, lgb_optimize_params, seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, get_config_name(), _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        print(f"    {in_trn_path}")
        print(f"    {in_tst_path}")
        return out_path, in_trn_path, in_tst_path

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    lgb_optimize_params['num_leaves'] = study.best_params['num_leaves']
    lgb_optimize_params['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    lgb_optimize_params['max_bin'] = study.best_params['max_bin']
    lgb_optimize_params['feature_fraction'] = study.best_params['feature_fraction']
    lgb_optimize_params['bagging_fraction'] = study.best_params['bagging_fraction']
    # params['lambda_l1'] = study.best_params['lambda_l1']
    lgb_optimize_params['learning_rate'] = 0.005

    with open(out_path, 'w') as fp:
        json.dump(lgb_optimize_params, fp)

    return out_path, in_trn_path, in_tst_path


knn_optimize_params = {
    'n_neighbors': 5,
    'weights': 'uniform',

}


def _302_optimize_knn(in_trn_path, in_tst_path, seed=CONST.SEED,
                      out_path=os.path.join(CONST.PIPE300, '_302_knn_optimized_params_seed{}_{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            knn_optimize_params['n_neighbors'] = trial.suggest_int('n_neighbors', 2, 32)
            knn_optimize_params['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])

            return knn_cv_id_fold(trn, knn_optimize_params, seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, get_config_name(), _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        print(f"    {in_trn_path}")
        print(f"    {in_tst_path}")
        return out_path, in_trn_path, in_tst_path

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    knn_optimize_params['n_neighbors'] = study.best_params['n_neighbors']
    knn_optimize_params['weights'] = study.best_params['weights']

    with open(out_path, 'w') as fp:
        json.dump(knn_optimize_params, fp)

    return out_path, in_trn_path, in_tst_path


optimization_func_mappter = {
    "lgb": _301_optimize_lgb,
    "knn": _302_optimize_knn
}


def _300_optimization(seed=CONST.SEED):
    trn_path, tst_path = _200_selection(seed)
    param_path, trn_path, tst_path = optimization_func_mappter[get_config()['_300_optimization']['func']](trn_path,
                                                                                                          tst_path,
                                                                                                          seed)

    return param_path, trn_path, tst_path


if __name__ == '__main__':
    param_path, trn_path, tst_path = _300_optimization(seed=42)
