'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples.

Command-line arguments:
1. type_context in {diff, sim, sem, size}
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
from testing import predict_model

_, type_context = sys.argv
data_partition = 'val_white'
path_weights = '/home/freddie/attention/weights/'
path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_splitdata = f'/home/freddie/ILSVRC2012-{type_context}contexts/{data_partition}/'
path_dataframes = f'/home/freddie/dataframes_{data_partition}/{type_context}contexts/'
path_initmodel = f'/home/freddie/keras-models/{type_context}contexts_initialised_model.h5'
path_results = '/home/freddie/attention/results/'
model = build_model(Attention(), train=False, attention_position=19)
print(f'Metrics: {model.metrics_names}')
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]

if type_source == 'directory':
    num_contexts = len(os.listdir(path_splitdata))
else:
    num_contexts = len(os.listdir(path_dataframes)) // 2

for i in range(num_contexts):
    name_context = f'{type_context}context{i:02}'
    print(f'\nTesting on {name_context}')
    W = np.load(f'{path_weights}{name_context}_weights_v5.npy')
    model.load_weights(path_initmodel) # `del model` deletes an existing model
    model.layers[ind_attention].set_weights([W])
    probabilites, generator = predict_model(model, type_source, path_data)
    np.save(f'{path_results}{name_context}_predictions.npy', probabilites, allow_pickle=False)
    np.save(f'{path_results}{name_context}_labels.npy', generator.classes, allow_pickle=False)
