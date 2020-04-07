"""
Sanity check: test an attention network with attention weights set to 1.
Agreement with the results produced by `vgg16_testing.py` implies that the
attention network works as expected.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
from ..utils.paths import path_imagenet, path_results
from ..utils.models import build_model
from ..utils.testing import evaluate_classwise_accuracy, predict_model

ind_attention = 19
model = build_model(train=False, attention_position=ind_attention)
predictions, generator = predict_model(
    model, 'dir', path_imagenet/data_partition)
df = evaluate_classwise_accuracy(predictions, generator)
df.to_csv(path_results/'untrained_attn_results.csv', index=False)
mean_acc = np.mean(df['accuracy'])
print(f'Mean accuracy on data partition = {mean_acc}')
