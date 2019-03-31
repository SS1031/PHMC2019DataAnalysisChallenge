import os
import pandas as pd
import datetime
import configparser
import json
import CONST


def get_config_name():
    inifile = configparser.ConfigParser()
    inifile.read('./conf.ini')
    return os.path.basename(inifile['conf']['configfile']).split('.')[0]


def get_config():
    inifile = configparser.ConfigParser()
    inifile.read('./conf.ini')

    with open(inifile['conf']['configfile'], "r") as fp:
        conf = json.load(fp)
    return conf


def update_result(pred_function_name, score):
    config_name = get_config_name()
    dt_now = datetime.datetime.now().replace(microsecond=0).isoformat()

    new_row = pd.DataFrame([[config_name, pred_function_name, dt_now, score]],
                           columns=['config', 'pred_func_name', 'exec time', 'score'])
    if os.path.exists(CONST.RESULT_SUMMARY):
        df = pd.read_csv(CONST.RESULT_SUMMARY)
    else:
        df = pd.DataFrame(columns=['config', 'pred_func_name', 'exec time', 'score'])

    df = pd.concat([df, new_row], axis=0)
    df.to_csv(CONST.RESULT_SUMMARY, index=False)
    print("=== Update result summary ==== ")
    print(df)


if __name__ == '__main__':
    print(get_config_name())
    print(get_config())
