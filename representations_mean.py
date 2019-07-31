'''
For each ImageNet class, find the mean VGG16 representation of images in it.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np

path_activations = '/home/freddie/activations/'
path_save = '/home/freddie/attention/npy/'
mean_activations = []

for i in range(1000):
    class_activations = np.load(
        f'{path_activations}class{i:04}_activations.npy')
    mean_activations.append(np.mean(class_activations, axis=0))

mean_activations = np.array(mean_activations)
np.save(f'{path_save}mean_activations', mean_activations, allow_pickle=False)
