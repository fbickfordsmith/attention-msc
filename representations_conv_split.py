import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

data_partition = 'train'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_activations = '/home/freddie/activations-conv/'
path_split = '/home/freddie/activations-conv-split/'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'

wnids = open(path_synsets).read().splitlines()
generator = ImageDataGenerator().flow_from_directory(directory=path_data)
class_filename = pd.Series(generator.filenames).str.split('/', expand=True)
df = pd.DataFrame()
df['filename'] = class_filename[1].str.split('.', expand=True)[0]
df['class'] = class_filename[0]

for i, wnid in enumerate(wnids):
    if i % 100 == 0:
        print(f'i = {i:04}')
    path_class = path_split + wnid
    os.makedirs(path_class)
    filenames = (df.loc[df['class']==wnid])['filename']
    activations = np.load(f'{path_activations}class{i:04}_activations_conv.npy')
    for f, a in zip(filenames, activations):
        np.save(f'{path_class}/{f}_conv5.npy', a, allow_pickle=False)

# # check
# s = 0
# for folder in os.listdir(path_split):
#     s += len(os.listdir(path_split+folder))
# print(f'Number of files = {s}')
