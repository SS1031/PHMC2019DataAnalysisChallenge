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
from sgd_cv import sgd_cv_id_fold
from svr_lin_cv import svr_lin_cv_id_fold
from lasso_cv import lasso_cv_id_fold
from _200_selection import _200_selection

lgb_optimization_space = {
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
                      out_path=os.path.join(CONST.PIPE300, '_301_lgb_optimized_params_seed{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            lgb_optimization_space['num_leaves'] = trial.suggest_int('num_leaves', 2, 128)
            lgb_optimization_space['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 10, 128)
            lgb_optimization_space['max_bin'] = trial.suggest_int('max_bin', 4, 256)
            lgb_optimization_space['feature_fraction'] = trial.suggest_uniform('feature_fraction', 0.7, 1.0)
            lgb_optimization_space['bagging_fraction'] = trial.suggest_uniform('bagging_fraction', 0.7, 1.0)
            lgb_optimization_space['bagging_freq'] = trial.suggest_int('bagging_freq', 1, 3)
            lgb_optimization_space['bagging_seed'] = seed
            lgb_optimization_space['feature_fraction_seed'] = seed
            lgb_optimization_space['seed'] = seed
            # params['lambda_l1'] = trial.suggest_uniform('lambda_l1', .0, .1)
            # params['lambda_l2'] = trial.suggest_uniform('lambda_l2', .0, .1)
            lgb_optimization_space['verbose'] = -1

            return lgb_cv_id_fold(trn, lgb_optimization_space, model_seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        return out_path, in_trn_path, in_tst_path

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    lgb_optimization_space['num_leaves'] = study.best_params['num_leaves']
    lgb_optimization_space['min_data_in_leaf'] = study.best_params['min_data_in_leaf']
    lgb_optimization_space['max_bin'] = study.best_params['max_bin']
    lgb_optimization_space['feature_fraction'] = study.best_params['feature_fraction']
    lgb_optimization_space['bagging_fraction'] = study.best_params['bagging_fraction']
    # params['lambda_l1'] = study.best_params['lambda_l1']
    # params['lambda_l2'] = study.best_params['lambda_l2']
    lgb_optimization_space['learning_rate'] = 0.005

    with open(out_path, 'w') as fp:
        json.dump(lgb_optimization_space, fp)

    return out_path, in_trn_path, in_tst_path


knn_optimization_space = {
    'n_neighbors': 5,
    'weights': 'uniform',
}


def _302_optimize_knn(in_trn_path, in_tst_path, seed=CONST.SEED,
                      out_path=os.path.join(CONST.PIPE300, '_302_knn_optimized_params_seed{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            knn_optimization_space['n_neighbors'] = trial.suggest_int('n_neighbors', 2, 32)
            knn_optimization_space['weights'] = trial.suggest_categorical('weights', ['uniform', 'distance'])

            return knn_cv_id_fold(trn, knn_optimization_space, model_seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        return out_path, in_trn_path, in_tst_path

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    knn_optimization_space['n_neighbors'] = study.best_params['n_neighbors']
    knn_optimization_space['weights'] = study.best_params['weights']

    with open(out_path, 'w') as fp:
        json.dump(knn_optimization_space, fp)

    return out_path, in_trn_path, in_tst_path


def _303_optimize_lin(in_trn_path, in_tst_path, seed=CONST.SEED,
                      out_path=os.path.join(CONST.PIPE300, '_303_lin_fake.json')):
    with open(out_path, 'w') as fp:
        json.dump({}, fp)

    return out_path, in_trn_path, in_tst_path


sgd_optimization_space = {
    'loss': 'squared_loss',
    'penalty': 'l2',
    'alpha': 0.0001,
    'fit_intercept': True,
    'max_iter': 10000,
    'tol': 1e-3,
    'verbose': 0,
    'early_stopping': True
}


def _304_optimize_sgd(in_trn_path, in_tst_path, seed=CONST.SEED,
                      out_path=os.path.join(CONST.PIPE300, '_304_sgd_optimized_params_seed{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            sgd_optimization_space['loss'] = trial.suggest_categorical('loss', ['squared_loss',
                                                                                'huber',
                                                                                'epsilon_insensitive',
                                                                                'squared_epsilon_insensitive'])
            sgd_optimization_space['penalty'] = trial.suggest_categorical('penalty', ['none', 'l2', 'l1', 'elasticnet'])
            sgd_optimization_space['alpha'] = trial.suggest_loguniform('alpha', 1e-5, 1e-3)

            return sgd_cv_id_fold(trn, sgd_optimization_space, model_seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        return out_path, in_trn_path, in_tst_path

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, _hash)

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)

    return out_path, in_trn_path, in_tst_path


svr_lin_optimization_space = {
    'kernel': 'linear',
    'C': 1.0,
}


def _305_optimize_svr_lin(in_trn_path, in_tst_path, seed=CONST.SEED,
                          out_path=os.path.join(CONST.PIPE300, '_305_svr_lin_optimized_params_seed{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn

            svr_lin_optimization_space['C'] = trial.suggest_loguniform('C', 1e-2, 1e2)

            return svr_lin_cv_id_fold(trn, svr_lin_optimization_space, model_seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        return out_path, in_trn_path, in_tst_path

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    svr_lin_optimization_space['C'] = study.best_params['C']

    with open(out_path, 'w') as fp:
        json.dump(svr_lin_optimization_space, fp)

    return out_path, in_trn_path, in_tst_path


lasso_optimization_space = {
    'alpha': 1.0,
    'fit_intercept': True,
    'normalize': False,
}


def _306_optimize_lasso(in_trn_path, in_tst_path, seed=CONST.SEED,
                        out_path=os.path.join(CONST.PIPE300, '_306_lasso_optimized_params_seed{}_{}.json')):
    class Objective(object):
        def __init__(self, trn):
            self.trn = trn

        def __call__(self, trial):
            trn = self.trn
            lasso_optimization_space['alpha'] = trial.suggest_uniform('alpha', 1e-2, 1)
            return lasso_cv_id_fold(trn, lasso_optimization_space, model_seed=CONST.SEED)

    _hash = hashlib.md5((in_trn_path + in_tst_path).encode('utf-8')).hexdigest()[:3]
    out_path = out_path.format(seed, _hash)

    if os.path.exists(out_path):
        print("Cache file exist")
        print(f"    {out_path}")
        return out_path, in_trn_path, in_tst_path

    trn = pd.read_feather(in_trn_path)

    objective = Objective(trn)
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    lasso_optimization_space['alpha'] = study.best_params['alpha']

    with open(out_path, 'w') as fp:
        json.dump(lasso_optimization_space, fp)

    return out_path, in_trn_path, in_tst_path


optimization_func_mappter = {
    "lgb": _301_optimize_lgb,
    "knn": _302_optimize_knn,
    "lin": _303_optimize_lin,
    "sgd": _304_optimize_sgd,
    "svr_lin": _305_optimize_svr_lin,
    "lasso": _306_optimize_lasso,
}


def _300_optimization(model, seed=CONST.SEED):
    trn_path, tst_path = _200_selection(seed)
    param_path, trn_path, tst_path = optimization_func_mappter[model](trn_path, tst_path, seed)

    return param_path, trn_path, tst_path


if __name__ == '__main__':
    param_path, trn_path, tst_path = _300_optimization('lasso')
