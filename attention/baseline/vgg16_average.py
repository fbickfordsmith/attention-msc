"""
Assess the average accuracy of a pretrained VGG16.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from ..utils.paths import path_imagenet
from ..utils.testing import evaluate_model

model = VGG16()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
scores = evaluate_model(model, 'directory', path_imagenet/data_partition)
print(f'{model.metrics_names} = {scores}')
