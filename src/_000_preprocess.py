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

    df.to_feather(save_path)

    return save_path


def _000_preprocess():
    trn_path = base_setup('trn')
    tst_path = base_setup('tst')

    return trn_path, tst_path


def _001_preprocess():
    """Regimeに分割したときに一定値のデータを削除する
    """
    out_trn_path = os.path.join(CONST.PIPE000, f'_001_trn.f')
    out_tst_path = os.path.join(CONST.PIPE000, f'_001_tst.f')

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    trn_path, tst_path = _000_preprocess()

    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    # 一定値のチェック
    tmp = (pd.concat([trn, tst], axis=0).groupby(['FlightRegime']).var() == 0).all(axis=0)
    regime_zero_variance_cols = tmp.index[tmp].values

    trn.drop(columns=regime_zero_variance_cols, inplace=True)
    tst.drop(columns=regime_zero_variance_cols, inplace=True)

    trn.to_feather(out_trn_path)
    tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


def _002_preprocess():
    """オフセット前処理
    """
    out_trn_path = os.path.join(CONST.PIPE000, f'_002_trn.f')
    out_tst_path = os.path.join(CONST.PIPE000, f'_002_tst.f')

    if os.path.exists(out_trn_path) and os.path.exists(out_tst_path):
        return out_trn_path, out_tst_path

    offset_features = [
        "T24TotalTemperatureAtLpcOutletR", "T30TotalTemperatureAtHpcOutletR",
        "T50TotalTemperatureAtLptOutletR", "NcPhysicalCoreSpeedRpm",
        "Ps30StaticPressureAtHpcOutletPsia", "NrcCorrectedCoreSpeedRpm", "BprBypassRatio",
        "HtbleedBleedEnthalpy", "W31HptCoolantBleedLbmS", "W32LptCoolantBleedLbmS",
    ]

    trn_path, tst_path = _001_preprocess()
    trn = pd.read_feather(trn_path)
    tst = pd.read_feather(tst_path)

    offset_trn = trn[offset_features + ['Engine', 'FlightNo', 'FlightRegime']].copy()
    offset_tst = tst[offset_features + ['Engine', 'FlightNo', 'FlightRegime']].copy()

    print("Train data offset")
    for f in offset_features:
        print("Offset Feature =", f)
        for eng in offset_trn.Engine.unique():
            for r in [1, 2, 3, 4, 5, 6]:
                offset_trn.loc[
                    (offset_trn.Engine == eng) & (offset_trn.FlightRegime == r), f
                ] -= offset_trn.loc[
                    (offset_trn.Engine == eng) & (offset_trn.FlightRegime == r), f
                ].iloc[0]

    print("Test data offset")
    for f in offset_features:
        print("Offset Feature =", f)
        for eng in offset_tst.Engine.unique():
            for r in [1, 2, 3, 4, 5, 6]:
                if len(offset_tst[(offset_tst.Engine == eng) & (offset_tst.FlightRegime == r)]) > 0:
                    offset_tst.loc[
                        (offset_tst.Engine == eng) & (offset_tst.FlightRegime == r), f
                    ] -= offset_tst.loc[
                        (offset_tst.Engine == eng) & (offset_tst.FlightRegime == r), f
                    ].iloc[0]

    rename_dict = dict(zip(offset_features, ['Offset' + f for f in offset_features]))

    offset_trn.rename(columns=rename_dict, inplace=True)
    offset_tst.rename(columns=rename_dict, inplace=True)

    offset_trn.to_feather(out_trn_path)
    offset_tst.to_feather(out_tst_path)

    return out_trn_path, out_tst_path


_000_mapper = {
    "drop_useless": _001_preprocess,
    "offset": _002_preprocess,
}

if __name__ == '__main__':
    trn_path, tst_path = _000_preprocess()
    # trn_path, tst_path = _001_preprocess()
    # trn_path, tst_path = _002_preprocess()
