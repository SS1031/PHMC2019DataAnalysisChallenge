import configparser
import json


def get_config():
    inifile = configparser.ConfigParser()
    inifile.read('./conf.ini')

    with open(inifile['conf']['configfile'], "r") as fp:
        conf = json.load(fp)
    return conf


if __name__ == '__main__':
    print(get_config())
