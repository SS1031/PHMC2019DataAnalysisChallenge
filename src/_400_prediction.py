import os
import json
import pandas as pd
import numpy as np

import CONST
import utils
from lgb_cv import lgb_n_fold_cv
from _300_optimization import _300_optimize


def _400_train_predict(seed=CONST.SEED):
    params_path, trn_path, tst_path = _300_optimize()

    with open(params_path, 'r') as fp:
        params = json.load(fp)

    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    score, preds = lgb_n_fold_cv(trn, params, tst, seed=seed)

    return score, preds


def _401_predict_weight1():
    func_name = '_401_predict_weight1'
    score, preds = _400_train_predict()
    sbmt = preds[preds.Weight == 1].groupby('Engine')[
        [c for c in preds.columns if 'fold' in c]
    ].mean().mean(axis=1).to_frame('Predicted RUL').reset_index()
    assert_engine = np.array(['Test' + str(i).zfill(3) for i in range(1, 101)]).astype(object)
    assert (sbmt['Engine'].values == assert_engine).all()
    utils.update_result(func_name, score)

    output_path = os.path.join(CONST.PIPE400, f'{func_name}_{utils.get_config_name()}.csv')
    sbmt[['Predicted RUL']].to_csv(output_path, index=False)
    return sbmt


def _402_seed_average_weight1(loops=10):
    func_name = '_402_seed_average_weight1'
    scores = []
    sbmts = pd.DataFrame()
    for i in range(1, loops):
        seed = CONST.SEED + i
        score, preds = _400_train_predict(seed)
        scores.append(score)
        _sbmt = preds[preds.Weight == 1].groupby('Engine')[
            [c for c in preds.columns if 'fold' in c]
        ].mean().mean(axis=1).to_frame(f'sbmt{i}')
        sbmts = pd.concat([sbmts, _sbmt], axis=1)

    utils.update_result(func_name, np.mean(score))
    sbmt = sbmts.mean(axis=1).to_frame('Predicted RUL').reset_index()
    output_path = os.path.join(CONST.PIPE400, f'{func_name}_{utils.get_config_name()}.csv')
    sbmt[['Predicted RUL']].to_csv(output_path, index=False)

    return sbmts


def _403_predict_weighted_average():
    func_name = '_401_predict_weight1'
    score, preds = _400_train_predict()
    # assert_engine = np.array(['Test' + str(i).zfill(3) for i in range(1, 101)]).astype(object)
    # assert (sbmt['Engine'].values == assert_engine).all()
    # utils.update_result(func_name, score)
    # output_path = os.path.join(CONST.PIPE400, f'{func_name}_{utils.get_config_name()}.csv')
    # sbmt[['Predicted RUL']].to_csv(output_path, index=False)
    return preds


if __name__ == '__main__':
    preds = _402_seed_average_weight1()
    # preds = _403_predict_weighted_average()
    # for col in [c for c in preds.columns if 'fold' in c]:
    #     preds[col] = preds[col] - preds['DiffFlightNo']
