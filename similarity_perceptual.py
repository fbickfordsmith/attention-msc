'''
For each pair of ImageNet classes, compute the similarity. Here, similarity is
measured by the perceptual similarity model of Zhang et al (2018) applied to
sampled examples from each class.

Needs to be moved into the PerceptualSimilarity folder.

References:
- github.com/richzhang/PerceptualSimilarity (see contents of `models/`)
- stackoverflow.com/questions/15208615/using-pth-files
'''


import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

from glob import glob
from itertools import product as cartesian
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import models
from util.util import im2tensor

# cp '/home/freddie/attention/similarity_perceptual.py' '/home/freddie/PerceptualSimilarity/similarity_perceptual.py'
data_partition = 'val_white'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_save = '/home/freddie/attention/activations/activations_perceptual.npy'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'

model = models.PerceptualLoss(
    model='net-lin', # model='net' => weight each feature equally
    net='vgg', # net='alex' => faster running
    use_gpu=True)

def compare_images(filepath0, filepath1):
    img0 = im2tensor(img_to_array(load_img(filepath0, target_size=(224, 224))))
    img1 = im2tensor(img_to_array(load_img(filepath1, target_size=(224, 224))))
    return float(model.forward(img0, img1))

import time
start = time.time()

dir0, dir1 = 'n01440764', 'n01440764' # 'n01943899'
paths_dir0 = glob(path_data+dir0+'/*')
paths_dir1 = glob(path_data+dir1+'/*')
path_combos = np.array(list(cartesian(paths_dir0, paths_dir1)))
distances = []
for i, (file0, file1) in enumerate(path_combos):
    if i % 500 == 0: print(i)
    distances.append(compare_images(file0, file1))
distances = np.array(distances)

print(time.time()-start)

# wnids = open(path_synsets).read().splitlines()
# pcpt_distances = np.load(path_save)
# ind_empty = np.transpose(np.nonzero(
#     (pcpt_distances-np.tril(np.ones_like(pcpt_distances), k=-1))==0))
# i0, j0 = ind_empty[0]
#
# for i in range(i0, 1000):
#     for j in range(j0, 1000):
#         if j >= i:
#             dir0, dir1 = wnids[i], wnids[j]
#             paths_dir0 = glob(path_data+dir0+'/*')
#             paths_dir1 = glob(path_data+dir1+'/*')
#             path_combos = np.array(list(cartesian(paths_dir0, paths_dir1)))
#             distances = []
#             for i, (filepath0, filepath1) in enumerate(path_combos):
#                 if i % 500 == 0: print(i)
#                 distances.append(compare_images(filepath0, filepath1))
#             pcpt_distances[i, j] = np.mean(np.array(distances))
#             np.save(path_save, pcpt_distances, allow_pickle=False)
