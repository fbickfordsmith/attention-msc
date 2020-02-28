"""
For each ImageNet category, assess the accuracy of a pretrained VGG16 on the
ImageNet validation set.
"""

gpu = input('GPU: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16
from ..utils.paths import path_imagenet, path_results
from ..utils.testing import predict_model

model = VGG16()
probabilities, generator = predict_model(
    model, 'directory', path_imagenet/'val_white/')
predictions = np.argmax(probabilities, axis=1)
labels = generator.classes
wnid2ind = generator.class_indices
correct_bool = (predictions == labels)
correct_class = labels[np.flatnonzero(correct_bool)] # vector where each entry is the class of an example that has been correctly classified

df = pd.DataFrame()
df['wnid'] = wnid2ind.keys()
df['num_examples'] = [np.count_nonzero(labels==i) for i in range(1000)]
df['num_correct'] = [np.count_nonzero(correct_class==i) for i in range(1000)]
df['accuracy'] = df['num_correct'] / df['num_examples']
df.to_csv(path_results/'base_results.csv')
