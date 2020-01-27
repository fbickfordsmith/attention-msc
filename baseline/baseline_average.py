"""
Assess the average accuracy of a pretrained VGG16 on the ImageNet validation
set.

References:
- medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os, sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.append('..')

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from utils.testing import evaluate_model
from utils.paths import path_imagenet

path_data = path_imagenet/data_partition

model = VGG16()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
scores = evaluate_model(model, 'directory', path_data)
print(f'{model.metrics_names} = {scores}')
