"""
Evaluate a pretrained VGG16 on ImageNet.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from ..utils.paths import path_imagenet, path_results
from ..utils.testing import evaluate_classwise_accuracy, predict_model

model = VGG16()
predictions, generator = predict_model(
    model, 'dir', path_imagenet/data_partition)
df = evaluate_classwise_accuracy(predictions, generator)
df.to_csv(path_results/'baseline_vgg16_results.csv', index=False)
mean_acc = np.mean(df['accuracy'])
print(f'Mean accuracy on data partition = {mean_acc}')
