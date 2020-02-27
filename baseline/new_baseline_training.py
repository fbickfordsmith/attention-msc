"""
Train an attention network on examples from all ImageNet categories.
"""

gpu = input('GPU: ')

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

ind_attention = 19
model = build_model(train=True, attention_position=ind_attention)
model, history = train_model(model, 'directory', path_data, use_data_aug=False)
pd.DataFrame(history.history).to_csv(path_training/f'baseline_training.csv')
np.save(
    path_weights/f'baseline_weights.npy',
    model.layers[ind_attention].get_weights()[0],
    allow_pickle=False)
