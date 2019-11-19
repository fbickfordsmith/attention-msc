'''
Assess the average accuracy of a pretrained VGG16 on the ImageNet validation
set.

References:
- medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
'''

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
from keras.applications.vgg16 import VGG16
from testing import evaluate_model

path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
scores = evaluate_model(model, 'directory', path_data)
print(f'{model.metrics_names} = {scores}')
