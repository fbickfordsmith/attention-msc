'''
For each ImageNet class, find the average VGG16 representation of images in it.
For each pair of classes, find the cosine similarity of the mean
representations.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

path_to_activations = '/home/freddie/activations/'
mean_activations = []

for i in range(1000):
    class_activations = np.load(
        path_to_activations+f'class{i:04}_activations.npy')
    mean_activations.append(np.mean(class_activations, axis=0))

mean_activations = np.array(mean_activations)
similarity = cosine_similarity(mean_activations)
np.save(path_to_activations+'mean_activations', mean_activations, allow_pickle=False)
np.save(path_to_activations+'cosine_similarity', similarity, allow_pickle=False)
