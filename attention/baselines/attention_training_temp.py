"""
Train a baseline attention network on examples from all ImageNet categories.
"""

gpu = input('GPU: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from ..utils.paths import path_imagenet, path_training, path_weights
from ..utils.models import build_model
from ..utils.training import train_model

ind_attention = 19
model = build_model(train=True, attention_position=ind_attention)
model, history = train_model(
    model, 'dir', path_imagenet/'train/', use_data_aug=False)
pd.DataFrame(history.history).to_csv(path_training/'baseline_attn_training_gpu1.csv')
np.save(
    path_weights/'baseline_attn_weights_gpu1.npy',
    model.layers[ind_attention].get_weights()[0],
    allow_pickle=False)
