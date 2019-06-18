import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
For each ImageNet class, get all VGG representations of the images in the class,
and take the mean of these.
For each pair of classes, find the cosine similarity of the mean representations.
'''

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
