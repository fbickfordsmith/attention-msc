'''
Assess the accuracy of a pretrained VGG16 on the ImageNet validation set.

Command-line arguments:
1. data_partition in {train, val, val_white}

References:
- medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from testing import evaluate_model

data_partition = input('Data partition: ')
# _, data_partition = sys.argv

path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
scores = evaluate_model(model, 'directory', path_data)
print(f'{model.metrics_names} = {scores}')
