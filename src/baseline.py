import os
import numpy as np
import pandas as pd
import CONST
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

trn_base = pd.read_csv(os.path.join(CONST.INDIR, 'concatenated_train.csv'))
tst_base = pd.read_csv(os.path.join(CONST.INDIR, 'concatenated_test.csv'))

trn_regime = pd.get_dummies(trn_base['Flight Regime'])
trn_regime.columns = [f'Regime{c}' for c in trn_regime.columns]
trn_base = pd.concat([trn_base, trn_regime], axis=1)

tst_regime = pd.get_dummies(tst_base['Flight Regime'])
tst_regime.columns = [f'Regime{c}' for c in tst_regime.columns]
tst_base = pd.concat([tst_base, tst_regime], axis=1)

feat_cols = ['Altitude', 'Mach #', 'Power Setting (TRA)',
             'T2 Total temperature at fan inlet ｰR',
             'T24 Total temperature at LPC outlet ｰR',
             'T30 Total temperature at HPC outlet ｰR',
             'T50 Total temperature at LPT outlet ｰR',
             'P2 Pressure at fan inlet psia',
             'P15 Total pressure in bypass-duct psia',
             'P30 Total pressure at HPC outlet psia', 'Nf Physical fan speed rpm',
             'Nc Physical core speed rpm', 'epr Engine pressure ratio (P50/P2) --',
             'Ps30 Static pressure at HPC outlet psia',
             'phi Ratio of fuel flow to Ps30 pps/psi', 'NRf Corrected fan speed rpm',
             'NRc Corrected core speed rpm', 'BPR Bypass Ratio --',
             'farB Burner fuel-air ratio --', 'htBleed (Bleed Enthalpy)',
             'Nf_dmd Demanded fan speed rpm',
             'PCNfR_dmd Demanded corrected fan speed rpm',
             'W31 HPT coolant bleed lbm/s', 'W32 LPT coolant bleed lbm/s',
             'Regime1', 'Regime2', 'Regime3', 'Regime4', 'Regime5', 'Regime6']

agg_dict = {}
for f in feat_cols:
    # agg_dict[f] = ['mean', 'median', 'min', 'max', 'sum', 'std',
    #                'var', 'sem', 'mad', 'skew', 'prod', pd.DataFrame.kurt]
    agg_dict[f] = ['mean']

trn_dataset = pd.DataFrame()
t_no_list = set(tst_base.groupby('Engine').FlightNo.max().values)
from tqdm import tqdm

for t_no in tqdm(t_no_list):
    t_engine = trn_base.groupby('Engine').size().index[trn_base.groupby('Engine').FlightNo.max() >= t_no].values
    tmp = trn_base[(trn_base.FlightNo <= t_no) & trn_base.Engine.isin(t_engine)]
    tmp_dataset = tmp.groupby(['Engine']).agg(agg_dict)
    tmp_dataset.columns = ['_'.join(col).strip() for col in tmp_dataset.columns.values]
    tmp_dataset['CurrentFlightNo'] = t_no
    tmp_dataset = pd.concat([
        tmp_dataset, trn_base[trn_base.Engine.isin(t_engine)].groupby(['Engine']).FlightNo.max().to_frame('RUL') - t_no
    ], axis=1).reset_index()  # Remaining Useful Life
    trn_dataset = pd.concat([trn_dataset, tmp_dataset], axis=0).reset_index(drop=True)

tst_dataset = tst_base.groupby('Engine').agg(agg_dict)
tst_dataset.columns = ['_'.join(col).strip() for col in tst_dataset.columns.values]
tst_dataset = pd.concat([tst_dataset, tst_base.groupby('Engine').FlightNo.max().to_frame('CurrentFlightNo')], axis=1)
tst_dataset = tst_dataset.reset_index()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
trn_dataset['EncodedEngine'] = le.fit_transform(trn_dataset['Engine'])

features = [c for c in trn_dataset.columns if c not in ['Engine', 'RUL', 'EncodedEngine']]
from sklearn.model_selection import GroupShuffleSplit

preds = pd.DataFrame({'Engine': tst_dataset.Engine, 'CurrentFlightNo': tst_dataset.CurrentFlightNo},
                     index=tst_dataset.index)
seed = 777
gsp = GroupShuffleSplit(n_splits=8, random_state=seed)
gsp.split(X=trn_dataset, groups=trn_dataset.EncodedEngine)

from sklearn.metrics import mean_absolute_error

mae_list = []
for ix, (train_index, valid_index) in enumerate(gsp.split(X=trn_dataset, groups=trn_dataset.EncodedEngine)):
    seed = seed * ix
    train_mean_rul = trn_dataset.loc[train_index, 'RUL'].mean()
    valid_rul = trn_dataset.loc[valid_index, ['RUL']].copy()
    valid_rul['Predicted RUL'] = train_mean_rul
    mae_list.append(mean_absolute_error(valid_rul.RUL, valid_rul['Predicted RUL']))

print(f"Average Baseline MAE = {np.mean(mae_list)}")
