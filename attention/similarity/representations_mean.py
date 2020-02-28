"""
For each ImageNet category, take the VGG16 representations of images in it
(computed using `representations_all.py`), and compute the mean of these.
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
from ..utils.paths import path_activations, path_representations

mean_activations = []

for i in range(1000):
    class_activations = np.load(
        path_activations/f'class{i:04}_activations.npy')
    mean_activations.append(np.mean(class_activations, axis=0))

np.save(
    path_representations/'representations_mean.npy',
    np.array(mean_activations),
    allow_pickle=False)
