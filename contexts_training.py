'''
ImageNet classes have been grouped into contexts. For each context, train an
attention layer on examples from that context only.

References:
- stackoverflow.com/questions/40496069/reset-weights-in-keras-layer/50257383
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import numpy as np
import pandas as pd
from models import build_model
from training import train_model

type_context = input('Context type in {diff, sem, sim, size}: ')
version_wnids = 'v' + input('Version number (WNIDs): ')
version_weights = 'v' + input('Version number (training/weights): ')
start = int(input('Start context: '))
stop = int(input('Stop context (inclusive): '))
data_partition = 'train'

path_weights = '/home/freddie/attention/weights/'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_dataframes = '/home/freddie/dataframes/'
path_initmodel = '/home/freddie/initialised_model.h5'
path_training = '/home/freddie/attention/training/'

model = build_model(train=True, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]

for i in range(start, stop+1):
    name_wnids = f'{type_context}_{version_wnids}_{i:02}'
    name_weights = f'{type_context}_{version_weights}_{i:02}'
    print(f'\nTraining on {name_wnids}')
    model.load_weights(path_initmodel)
    args_train = [pd.read_csv(f'{path_dataframes}{name_wnids}_df.csv'), path_data]
    model, history = train_model(model, 'dataframe', *args_train, use_data_aug=False)
    pd.DataFrame(history.history).to_csv(f'{path_training}{name_weights}_training.csv')
    np.save(
        f'{path_weights}{name_weights}_weights.npy',
        model.layers[ind_attention].get_weights()[0],
        allow_pickle=False)
