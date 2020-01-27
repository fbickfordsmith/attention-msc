"""
For each context, make a dataframe containing filepaths, labels and filenames
for all the examples in the context.
"""

gpu = input('GPU: ')
type_context = input('Context type in {diff, sem, sim, size}: ')
version_wnids = input('Version number (WNIDs): ')

import os, sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.append('..')

import csv
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.paths import path_repo, path_imagenet, path_dataframes

path_data = path_imagenet/'train/'
path_load = path_repo/f'data/contexts/{type_context}_v{version_wnids}_wnids.csv'

generator = ImageDataGenerator().flow_from_directory(directory=path_data)

df = pd.DataFrame({
    'filename': generator.filenames,
    'class': pd.Series(generator.filenames).str.split('/', expand=True)[0]})

with open(path_load) as f:
    contexts = [row for row in csv.reader(f, delimiter=',')]

for i, context in enumerate(contexts):
    inds_incontext = []
    for wnid in context:
        inds_incontext.extend(np.flatnonzero(df['class']==wnid))
    df.iloc[inds_incontext].to_csv(
        path_dataframes/f'{type_context}_v{version_wnids}_{i:02}_df.csv',
        index=False)
