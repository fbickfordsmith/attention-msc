"""
Test a baseline attention network on ImageNet.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from ..utils.paths import path_imagenet, path_results, path_weights
from ..utils.models import build_model
from ..utils.testing import predict_model

ind_attention = 19
model = build_model(train=False, attention_position=ind_attention)
weights = np.load(path_weights/'baseline_attn_weights.npy')
model.layers[ind_attention].set_weights([weights])
predictions, generator = predict_model(
    model, 'dir', path_imagenet/data_partition)
df = evaluate_classwise_accuracy(predictions, generator)
df.to_csv(path_results/'attn_baseline_results.csv', index=False)
mean_acc = np.mean(df['accuracy'])
print(f'Mean accuracy on data partition = {mean_acc}')
