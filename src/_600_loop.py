import pandas as pd
from _500_ensemble import _502_weighted_average


def _600_loop(seeds=[42, 777]):
    sbmts = pd.DataFrame()
    for seed in seeds:
        scores, preds, sbmt = _502_weighted_average(seed)
        sbmts = pd.concat([sbmts, sbmt.to_frame('SEED{}'.format(seed))], axis=1)

    return sbmts


if __name__ == '__main__':
    sbmts = _600_loop()
