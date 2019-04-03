import numpy as np
import pandas as pd
import featuretools as ft
import sandbox.featuretools_utils as utils

utils.download_data()
data_path = '../data/sandbox/train_FD004.txt'
data = utils.load_data(data_path)
print(data.head())

cutoff_times = utils.make_cutoff_times(data)
print("#############")
print("Cut off times")
print("#############")
print(cutoff_times.head())


def make_entityset(data):
    es = ft.EntitySet('Dataset')
    es.entity_from_dataframe(dataframe=data,
                             entity_id='recordings',
                             index='index',
                             time_index='time')

    es.normalize_entity(base_entity_id='recordings',
                        new_entity_id='engines',
                        index='engine_no')

    es.normalize_entity(base_entity_id='recordings',
                        new_entity_id='cycles',
                        index='time_in_cycles')
    return es


es = make_entityset(data)
print(es)

fm, features = ft.dfs(entityset=es,
                      target_entity='engines',
                      agg_primitives=['last', 'max', 'min'],
                      trans_primitives=[],
                      cutoff_time=cutoff_times,
                      max_depth=3,
                      verbose=True)
