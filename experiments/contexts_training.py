"""
For each context, train an attention network on examples from that context only.

References:
- stackoverflow.com/questions/40496069/reset-weights-in-keras-layer/50257383
"""

gpu = input('GPU: ')
type_context = input('Context type in {diff, sem, sim, size}: ')
version_wnids = input('Version number (WNIDs): ')
version_weights = input('Version number (training/weights): ')
start = int(input('Start context: '))
stop = int(input('Stop context (inclusive): '))

import os, sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.append('..')

import numpy as np
import pandas as pd
from utils.paths import *
from utils.models import build_model
from utils.training import train_model

path_weights = path_repo/'data/weights/'
path_data = path_imagenet/'train/'
path_training = path_repo/'data/training/'

model = build_model(train=True, attention_position=19)
model.save_weights(path_init_model)
ind_attention = np.flatnonzero(
    ['attention' in layer.name for layer in model.layers])[0]

for i in range(start, stop+1):
    name_wnids = f'{type_context}_v{version_wnids}_{i:02}'
    name_weights = f'{type_context}_v{version_weights}_{i:02}'
    print(f'\nTraining on {name_wnids}')
    model.load_weights(path_init_model)
    args_train = [
        pd.read_csv(path_dataframes/f'{name_wnids}_df.csv'), path_data]
    model, history = train_model(
        model, 'dataframe', *args_train, use_data_aug=False)
    pd.DataFrame(history.history).to_csv(
        path_training/f'{name_weights}_training.csv')
    np.save(
        path_weights/f'{name_weights}_weights.npy',
        model.layers[ind_attention].get_weights()[0],
        allow_pickle=False)
