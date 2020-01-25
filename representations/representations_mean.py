'''
For each ImageNet class, take the VGG16 representations of images in it
(computed using `representations_all.py`), and compute the mean of these.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import numpy as np

path_activations = '/home/freddie/activations/'
path_save = '/home/freddie/attention/representations/'
mean_activations = []

for i in range(1000):
    class_activations = np.load(
        f'{path_activations}class{i:04}_activations.npy')
    mean_activations.append(np.mean(class_activations, axis=0))

mean_activations = np.array(mean_activations)
np.save(f'{path_save}mean_activations', mean_activations, allow_pickle=False)
