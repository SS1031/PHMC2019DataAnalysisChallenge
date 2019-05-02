import os
import numpy as np
import pandas as pd
import CONST

import utils

from _400_prediction import _400_prediction


def _501_seed_average(model, loops=10, seed=CONST.SEED):
    print(f"Start Seed Averaging {model}...")
    scores = []
    sbmts = pd.DataFrame()

    for i in range(1, loops + 1):
        model_seed = seed * i
        print(f"Loop : {i}, Model Seed : {model_seed}, Cutoff Seed {seed}")
        score, preds = _400_prediction(model, model_seed=model_seed, co_seed=seed)
        print("CV Score :", score)
        scores.append(score)
        _sbmt = preds.mean(axis=1).to_frame(f'sbmt{i}')
        sbmts = pd.concat([sbmts, _sbmt], axis=1)

    sbmt = sbmts.mean(axis=1).to_frame('PredRUL_{}'.format(model)).reset_index(drop=True)
    return sbmt, np.mean(scores)


def _502_weighted_average(seed=CONST.SEED):
    """
    Weightの決め方は，
    - 各モデルのCVスコアの総和のそれぞれの割合の逆数
    例: model1 score = 1, model2 score = 2のとき

    weight1 = 1 / (1 / (1+2)) = 3
    weight2 = 1 / (2 / (1+2)) = 1.5
    """
    scores = []
    preds = pd.DataFrame()
    models = utils.get_config()['_500_ensemble']['models']
    for model in models:
        _pred, _score = _501_seed_average(model, loops=10, seed=seed)
        scores.append(_score)
        preds = pd.concat([preds, _pred], axis=1)

    print(preds.corr())
    scores = np.array(scores)
    weights = 1 / (scores / scores.sum())

    sbmt = (preds * weights).sum(axis=1) / weights.sum()

    return scores, preds, sbmt


if __name__ == '__main__':
    scores, preds, sbmt = _502_weighted_average()
