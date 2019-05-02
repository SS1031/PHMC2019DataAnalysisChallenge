import os
import json
import pandas as pd
import numpy as np

import CONST
import utils
from lgb_cv import lgb_cv_id_fold
from knn_cv import knn_cv_id_fold
from lin_cv import lin_cv_id_fold
from sgd_cv import sgd_cv_id_fold
from svr_lin_cv import svr_lin_cv_id_fold
from lasso_cv import lasso_cv_id_fold
from utils import get_config
from _300_optimization import _300_optimization

predict_func_mapper = {
    'lgb': lgb_cv_id_fold,
    'knn': knn_cv_id_fold,
    'lin': lin_cv_id_fold,
    'sgd': sgd_cv_id_fold,
    'svr_lin': svr_lin_cv_id_fold,
    'lasso': lasso_cv_id_fold,
}


def _400_prediction(model, model_seed=CONST.SEED, co_seed=CONST.SEED):
    params_path, trn_path, tst_path = _300_optimization(model, co_seed)

    with open(params_path, 'r') as fp:
        params = json.load(fp)

    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    score, preds = predict_func_mapper[model](trn, params=params, tst=tst, model_seed=model_seed)

    return score, preds


def _401_seed_average(loops=10):
    print("Start Seed Averaging ...")
    func_name = "seed_average"
    output_path = os.path.join(CONST.PIPE400, f'{utils.get_config_name()}.csv')
    scores = []
    sbmts = pd.DataFrame()

    for i in range(1, loops + 1):
        seed = CONST.SEED * i
        print(f"Loop : {i}, Seed : {seed}")
        score, preds = _400_prediction(get_config()['model'], seed)
        print("CV Score :", score)
        scores.append(score)
        _sbmt = preds.mean(axis=1).to_frame(f'sbmt{i}')
        sbmts = pd.concat([sbmts, _sbmt], axis=1)

    utils.update_result(func_name, np.mean(scores), np.std(scores), output_path)

    return sbmts


if __name__ == '__main__':
    preds = _401_seed_average(loops=50)
