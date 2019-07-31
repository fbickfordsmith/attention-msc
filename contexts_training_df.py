'''
ImageNet classes have been grouped into contexts. For each context, train an
attention layer on examples from that context only.

Command-line arguments:
1. type_context in {diff, sim, sem, size}

References:
- stackoverflow.com/questions/40496069/reset-weights-in-keras-layer/50257383
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import numpy as np
import pandas as pd
from layers import Attention
from models import build_model
from training_df import train_model

_, type_context = sys.argv
path_weights = '/home/freddie/attention/weights/'
path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
path_dataframes = f'/home/freddie/dataframes_train/{type_context}contexts/'
path_initmodel = f'/home/freddie/keras-models/{type_context}contexts_initialised_model.h5'
path_training = '/home/freddie/attention/training/'
model = build_model(Attention(), train=True)
model.save_weights(path_initmodel)
num_contexts = len(os.listdir(path_dataframes))

for i in range(num_contexts):
    context_name = f'{type_context}context{i:02}'
    print(f'\nTraining on {context_name}')
    dataframe = pd.read_csv(f'{path_dataframes}context{i:02}_dataframe.csv')
    model.load_weights(path_initmodel)
    model, history = train_model(model, dataframe, path_data)
    ind_attention = np.flatnonzero(
        ['attention' in layer.name for layer in model.layers])[0]
    pd.DataFrame(history.history).to_csv(
        f'{path_training}{context_name}_training.csv')
    np.save(
        f'{path_weights}{context_name}_attention_weights.npy',
        model.layers[ind_attention].get_weights()[0],
        allow_pickle=False)
