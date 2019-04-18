import os
import re
import pandas as pd

import CONST

regex = re.compile('[^a-zA-Z0-9]')


def base_setup(kind='trn'):
    """
    :param kind: "trn" or "tst"
    """
    save_path = os.path.join(CONST.PIPE000, f'_000_{kind}.f')

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

    df.columns = ["".join(regex.sub('', c.title()).split(' ')).strip() for c in df.columns]

    df_regime = pd.get_dummies(df['FlightRegime'])
    df_regime.columns = [f'Regime{c}' for c in df_regime.columns]
    df = pd.concat([df, df_regime], axis=1)
    df = df.drop(columns=['FlightRegime'])

    df.to_feather(save_path)

    return save_path


def _000_preprocess():
    trn_path = base_setup('trn')
    tst_path = base_setup('tst')

    return trn_path, tst_path


def _001_preprocess():
    """Regimeごとに変化がないデータを削除する

    """
    out_trn_path = os.path.join(CONST.PIPE000, f'_001_trn.f')
    out_tst_path = os.path.join(CONST.PIPE000, f'_001_tst.f')

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_path, tst_path = _000_preprocess()

    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    drop_cols = ["Altitude",
                 "Mach",
                 "PowerSettingTra",
                 "T2TotalTemperatureAtFanInletR",
                 "P2PressureAtFanInletPsia",
                 "P15TotalPressureInBypassDuctPsia",
                 "FarbBurnerFuelAirRatio",
                 "NfDmdDemandedFanSpeedRpm"]

    trn.drop(columns=drop_cols, inplace=True)
    tst.drop(columns=drop_cols, inplace=True)

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


if __name__ == '__main__':
    trn_path, tst_path = _001_preprocess()
