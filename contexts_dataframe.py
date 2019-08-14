'''
For each context, make a dataframe containing filepaths, labels and filenames
for all the examples in the context.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import sys
import csv
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

type_context = input('Context type in {diff, sem, sim, size}: ')
version = input('Version number: ')
data_partition = 'train'

path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_contexts = '/home/freddie/attention/contexts/'
path_dataframes = '/home/freddie/dataframes/'

generator = ImageDataGenerator().flow_from_directory(directory=path_data)
df = pd.DataFrame()
df['filename'] = generator.filenames
df['class'] = pd.Series(generator.filenames).str.split('/', expand=True)[0]

with open(f'{path_contexts}{type_context}contexts_wnids_v{version}.csv') as f:
    contexts = [row for row in csv.reader(f, delimiter=',')]

for i, context in enumerate(contexts):
    inds_incontext = []
    for wnid in context:
        inds_incontext.extend(np.flatnonzero(df['class']==wnid))

    df.iloc[inds_incontext].to_csv(
        f'{path_dataframes}{type_context}context{i:02}_df_v{version}.csv', index=False)
