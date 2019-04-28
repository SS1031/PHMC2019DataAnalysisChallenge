import os
import json
import pandas as pd
import numpy as np

import CONST
import utils
from lgb_cv import lgb_cv_id_fold
from _300_optimization import _301_optimize_cv_id


def _400_train_predict(seed=CONST.SEED):
    params_path, trn_path, tst_path = _301_optimize_cv_id()

    with open(params_path, 'r') as fp:
        params = json.load(fp)

    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    score, preds = lgb_cv_id_fold(trn, params=params, tst=tst, seed=seed)

    return score, preds


def _402_seed_average(loops=10):
    func_name = '_402_seed_average'
    output_path = os.path.join(CONST.PIPE400, f'{func_name}_{utils.get_config_name()}.csv')
    scores = []
    sbmts = pd.DataFrame()
    for i in range(1, loops + 1):
        seed = CONST.SEED * i
        print("Loop :", i, "Seed :", seed)
        score, preds = _400_train_predict(seed)
        print("CV Score :", score)
        scores.append(score)
        _sbmt = preds.mean(axis=1).to_frame(f'sbmt{i}')
        sbmts = pd.concat([sbmts, _sbmt], axis=1)

    utils.update_result(func_name, np.mean(scores), output_path)
    sbmt = sbmts.mean(axis=1).to_frame('Predicted RUL').reset_index()
    sbmt[['Predicted RUL']].to_csv(output_path, index=False)

    return sbmts


if __name__ == '__main__':
    preds = _402_seed_average()
