'''
ImageNet classes have been grouped into contexts. For each context, train an
attention layer on examples from that context only.

Command-line arguments:
1. type_context in {diff, sim, sem, size}
2. type_source in {directory, dataframe}

References:
- stackoverflow.com/questions/40496069/reset-weights-in-keras-layer/50257383
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import numpy as np
import pandas as pd
from models import build_model
from training import train_model

_, type_context, type_source = sys.argv
data_partition = 'train'
path_weights = '/home/freddie/attention/weights/'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_splitdata = f'/home/freddie/ILSVRC2012-{type_context}contexts/{data_partition}/'
path_dataframes = f'/home/freddie/dataframes_{data_partition}/{type_context}contexts/'
path_initmodel = f'/home/freddie/keras-models/{type_context}contexts_initialised_model.h5'
path_training = '/home/freddie/attention/training/'
model = build_model(train=True, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]
if type_source == 'directory':
    num_contexts = len(os.listdir(path_splitdata))
else:
    num_contexts = len(os.listdir(path_dataframes))

for i in range(num_contexts):
    name_context = f'{type_context}context{i:02}'
    print(f'\nTraining on {name_context}')
    model.load_weights(path_initmodel)
    if type_source == 'directory':
        args_train = [f'{path_splitdata}context{i:02}/']
    elif type_source == 'dataframe':
        args_train = [pd.read_csv(f'{path_dataframes}{name_context}_df.csv'), path_data]
    else:
        raise ValueError(f'Invalid value for type_source: {type_source}')
    model, history = train_model(model, type_source, *args_train, use_data_aug=False)
    pd.DataFrame(history.history).to_csv(f'{path_training}{name_context}_training.csv')
    np.save(
        f'{path_weights}{name_context}_weights.npy',
        model.layers[ind_attention].get_weights()[0],
        allow_pickle=False)
