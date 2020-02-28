"""
For each category set, train an attention network on examples from that category
set only.
"""

gpu = input('GPU: ')
type_category_set = input('Category-set type in {diff, sem, sim, size}: ')
version_wnids = input('Version number (WNIDs): ')
version_weights = input('Version number (training/weights): ')
start = int(input('Start category set: '))
stop = int(input('Stop category set (inclusive): '))

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from ..utils.paths import (path_dataframes, path_imagenet, path_init_model,
    path_training, path_weights)
from ..utils.models import build_model
from ..utils.training import train_model

ind_attention = 19
model = build_model(train=True, attention_position=ind_attention)
model.save_weights(path_init_model)

for i in range(start, stop+1):
    name_wnids = f'{type_category_set}_v{version_wnids}_{i:02}'
    name_weights = f'{type_category_set}_v{version_weights}_{i:02}'
    print(f'\nTraining on {name_wnids}')
    model.load_weights(path_init_model)
    args_train = [
        pd.read_csv(path_dataframes/f'{name_wnids}_df.csv'),
        path_imagenet/'train/']
    model, history = train_model(model, 'df', *args_train, use_data_aug=False)
    pd.DataFrame(history.history).to_csv(
        path_training/f'{name_weights}_training.csv')
    np.save(
        path_weights/f'{name_weights}_weights.npy',
        model.layers[ind_attention].get_weights()[0],
        allow_pickle=False)
