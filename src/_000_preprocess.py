import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import CONST

regex = re.compile('[^a-zA-Z]')

plt.style.use('seaborn')

files_trn = os.listdir(CONST.INTRNDIR)
files_tst = os.listdir(CONST.INTSTDIR)
trn = pd.DataFrame()
for f in files_trn:
    df = pd.read_csv(os.path.join(CONST.INTRNDIR, f), encoding='shift-jis', usecols=list(range(0, 25)))
    df['Engine'] = 'Train' + f.split('.')[0].split('_')[2]
    df['Flight No'] = df.index.values + 1
    trn = pd.concat([trn, df], axis=0).reset_index(drop=True)

trn.columns = ["".join(regex.sub('', c.title()).split(' ')) for c in trn.columns]

trn.to_csv(os.path.join(CONST.INDIR, 'trn_base.csv'), index=False)

tst = pd.DataFrame()
for f in files_tst:
    df = pd.read_csv(os.path.join(CONST.INTSTDIR, f), encoding='shift-jis', usecols=list(range(0, 25)))
    df['Engine'] = 'Test' + f.split('.')[0].split('_')[2]
    df['Flight No'] = df.index.values + 1
    tst = pd.concat([tst, df], axis=0).reset_index(drop=True)
tst.columns = ["".join(regex.sub('', c.title()).split(' ')) for c in tst.columns]
tst.to_csv(os.path.join(CONST.INDIR, 'tst_base.csv'), index=False)

