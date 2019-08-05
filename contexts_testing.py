'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples.

Command-line arguments:
1. type_context in {diff, sim, sem, size}
2. type_source in {directory, dataframe}
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import itertools
import numpy as np
import pandas as pd
from layers import Attention
from models import build_model
from testing import evaluate_model

_, type_context, type_source = sys.argv
data_partition = 'val_white'
path_weights = '/home/freddie/attention/weights/'
path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_splitdata = f'/home/freddie/ILSVRC2012-{type_context}contexts/{data_partition}/'
path_dataframes = f'/home/freddie/dataframes_{data_partition}/{type_context}contexts/'
path_initmodel = f'/home/freddie/keras-models/{type_context}contexts_initialised_model.h5'
path_results = '/home/freddie/attention/results/'
model = build_model(Attention(), train=False, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]
scores_ic, scores_ooc = [], []

if type_source == 'directory':
    num_contexts = len(os.listdir(path_splitdata))
else:
    num_contexts = len(os.listdir(path_dataframes)) // 2

for i in range(num_contexts):
    name_context = f'{type_context}context{i:02}'
    print(f'\nTesting on {name_context}')
    weights = np.load(f'{path_weights}{name_context}_weights_v5.npy')
    # weights = np.load(f'{path_weights}{name_context}_weights.npy')
    model.load_weights(path_initmodel) # `del model` deletes an existing model
    model.layers[ind_attention].set_weights([weights])

    if type_source == 'directory':
        scores_ic.append(evaluate_model(model, type_source, f'{path_splitdata}context{i:02}/'))
        scores_temp = [
            evaluate_model(model, type_source, f'{path_splitdata}context{j:02}/')
            for j in range(num_contexts) if j != i]
        scores_ooc.append(np.mean(np.array(scores_temp), axis=0))
    elif type_source == 'dataframe':
        df_ic = pd.read_csv(f'{path_dataframes}{name_context}_df.csv')
        df_ooc = pd.read_csv(f'{path_dataframes}{name_context}_df_out.csv')
        scores_ic.append(evaluate_model(model, type_source, df_ic, path_data))
        scores_ooc.append(evaluate_model(model, type_source, df_ooc, path_data))
    else:
        raise ValueError(f'Invalid value for type_source: {type_source}')

scores_arr = np.concatenate((np.array(scores_ic), np.array(scores_ooc)), axis=1)

# list(itertools.chain(list1, list2, ...)) returns a flattened list
col_names = list(itertools.chain(
    [f'incontext_{metric_name}' for metric_name in model.metrics_names],
    [f'outofcontext_{metric_name}' for metric_name in model.metrics_names]))

pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'{path_results}{type_context}contexts_trained_metrics.csv')
