import os
import pandas as pd
from _500_ensemble import _502_weighted_average

import utils

if not os.path.exists('../data/sbmts'):
    os.mkdir('../data/sbmts')


def _600_loop(seeds=[42, 777]):
    sbmts = pd.DataFrame()
    for seed in seeds:
        scores, preds, sbmt = _502_weighted_average(seed)
        sbmts = pd.concat([sbmts, sbmt.to_frame('SEED{}'.format(seed))], axis=1)
    return sbmts


if __name__ == '__main__':
    output_path = os.path.join('../data/sbmts', f'{utils.get_config_name()}.csv')
    sbmts = _600_loop(seeds=[21, 23, 33, 35, 42, 777, 999])
    sbmts.mean(axis=1).to_frame('Predicted RUL')
    sbmt = sbmts.mean(axis=1).to_frame('Predicted RUL').reset_index()
    # Post-Processing...
    sbmt.loc[sbmt['Predicted RUL'] < 0, 'Predicted RUL'] = 10
    sbmt[['Predicted RUL']].to_csv(output_path, index=False)
