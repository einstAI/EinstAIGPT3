# -*- coding: utf-8 -*-

import configparser
import json


class DictParser(configparser.ConfigParser):
    def read_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


cf = DictParser()
cf.read("config.ini", encoding="utf-8")
config_dict = cf.read_dict()


def parse_args():
    return config_dict["database_tune"]


def parse_ricci_config():
    _ricci_config = config_dict["ricci_config"]
    for key in _ricci_config:
        _ricci_config[key] = json.loads(str(_ricci_config[key]).replace("\'", "\""))
    return _ricci_config


ricci_config = parse_ricci_config()
# sync with main.py

predictor_output_dim = int(config_dict["predictor"]["predictor_output_dim"])

predictor_epoch = int(config_dict["predictor"]["predictor_epoch"])
