import os
import re
import pandas as pd

import CONST

regex = re.compile('[^a-zA-Z]')


def base_setup(kind='trn'):
    """
    :param kind: "trn" or "tst"
    """
    save_path = os.path.join(CONST.PIPE000, f'{kind}_base.csv')

    if os.path.exists(save_path):
        return save_path

    if kind == 'trn':
        dir_path = CONST.INTRNDIR
    if kind == 'tst':
        dir_path = CONST.INTSTDIR

    files = os.listdir(dir_path)
    df = pd.DataFrame()
    for f in files:
        _df = pd.read_csv(os.path.join(dir_path, f), encoding='shift-jis',
                          usecols=list(range(0, 25)))
        if kind == 'trn':
            _df['Engine'] = 'Train' + f.split('.')[0].split('_')[2]
        else:
            _df['Engine'] = 'Test' + f.split('.')[0].split('_')[2]
        _df['Flight No'] = _df.index.values + 1
        df = pd.concat([df, _df], axis=0).reset_index(drop=True)

    df.columns = ["".join(regex.sub('', c.title()).split(' ')) for c in df.columns]

    df_regime = pd.get_dummies(df['FlightRegime'])
    df_regime.columns = [f'Regime{c}' for c in df_regime.columns]
    df = pd.concat([df, df_regime], axis=1)
    df = df.drop(columns=['FlightRegime'])

    df.to_csv(save_path, index=False)

    return save_path


def _000_preprocess():
    trn_path = base_setup('trn')
    tst_path = base_setup('tst')

    return trn_path, tst_path


if __name__ == '__main__':
    trn_path, tst_path = _000_preprocess()
