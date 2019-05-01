import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import Lasso

from sklearn.metrics import mean_absolute_error

import CONST
import utils


def lasso_cv_id_fold(trn, params={}, tst=None, model_seed=CONST.SEED):
    """
    """
    if tst is not None:
        preds = tst[['Engine']].copy()

    cv_id = utils.get_cv_id(model_seed)
    trn = trn.merge(cv_id, on=['Engine'], how='left')
    assert trn.cv_id.notnull().all()

    valid_preds = pd.DataFrame({'preds': [np.nan] * trn.shape[0], 'actual_RUL': trn.RUL})
    features = [c for c in trn.columns if c not in CONST.EX_COLS]

    scaler = preprocessing.StandardScaler()
    trn.loc[:, features] = scaler.fit_transform(trn.loc[:, features])
    if tst is not None:
        tst.loc[:, features] = scaler.transform(tst.loc[:, features])

    for i in list(range(1, utils.get_config()['nfold'] + 1)):
        print(f"CV ID = {i}")
        X_train, y_train = trn.loc[trn.cv_id != i, features], trn.loc[trn.cv_id != i, 'RUL']
        X_valid, y_valid = trn.loc[trn.cv_id == i, features], trn.loc[trn.cv_id == i, 'RUL']

        model = Lasso(**params, random_state=model_seed)
        model.fit(X_train, y_train)
        valid_preds.loc[trn.cv_id == i, 'preds'] = model.predict(X_valid)

        if tst is not None:
            preds[f'fold{i + 1}'] = model.predict(tst[features])

    valid_preds.dropna(inplace=True)
    if tst is None:
        print("CV MAE Score :", mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds))
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds)
    else:
        return mean_absolute_error(valid_preds.actual_RUL, valid_preds.preds), preds
