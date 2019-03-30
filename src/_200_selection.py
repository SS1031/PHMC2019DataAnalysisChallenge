import os
import CONST
import pandas as pd

from _100_feature import _100_feature
from utils import get_config


def _201_drop_zero_variance(input_path, output_path=os.path.join(CONST.PIPE200,
                                                                 '_201_drop_zero_std_tsfeaure.f')):
    if os.path.exists(output_path):
        return output_path

    dataset = pd.read_feather(input_path)

    print("Before drop zero variance features,", dataset.shape)
    cols_std = dataset.std()
    drop_features = cols_std[cols_std == 0].index.values
    dataset = dataset.drop(columns=drop_features)
    print("After drop zero variance features,", dataset.shape)

    dataset.to_feather(output_path)

    return output_path


def lgb_top_100(path):
    return path


mapper = {
    "drop_zero_variance": _201_drop_zero_variance(),
    "lgb_top_": lgb_top_k,
}


def _200_selection():
    _path = _100_feature()
    for selection in get_config()['_002_selection']:
        func_selection = mapper[selection]
        _path = func_selection(_path)
