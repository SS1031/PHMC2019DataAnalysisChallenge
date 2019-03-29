import CONST
from _000_preprocess import _000_preprocess
from utils import get_config
from _100_feature import _100_feature


def drop_zero_variance(path):
    return path


def lgb_top_k(path):
    return path


mapper = {
    "drop_zero_variance": drop_zero_variance,
    "lgb_top_": lgb_top_k,
}


def _200_selection():
    _path = _100_feature()
    for selection in get_config()['_002_selection']:
        func_selection = mapper[selection]
        _path = func_selection(_path)
