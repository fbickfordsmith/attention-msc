"""
For each category set, make a dataframe containing filepaths, labels and
filenames for all the examples in the category set.
"""

gpu = input('GPU: ')
type_category_set = input('Category-set type in {diff, sem, sim, size}: ')
version_wnids = input('Version number (WNIDs): ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import csv
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ..utils.paths import path_category_sets, path_dataframes, path_imagenet

path_data = path_imagenet/'train/'
path_load = path_category_sets/f'{type_category_set}_v{version_wnids}_wnids.csv'

generator = ImageDataGenerator().flow_from_directory(directory=path_data)

df = pd.DataFrame({
    'filename': generator.filenames,
    'class': pd.Series(generator.filenames).str.split('/', expand=True)[0]})

with open(path_load) as f:
    category_sets = [row for row in csv.reader(f, delimiter=',')]

for i, category_set in enumerate(category_sets):
    inds_in_set = []
    for wnid in category_set:
        inds_in_set.extend(np.flatnonzero(df['class']==wnid))
    df.iloc[inds_in_set].to_csv(
        path_dataframes/f'{type_category_set}_v{version_wnids}_{i:02}_df.csv',
        index=False)
