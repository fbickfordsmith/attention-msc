"""
Define a set of 6 semantic category sets. These are subsets of ImageNet
categories that are conceptually similar, as judged by humans.

File in `data/category_sets/`     Number of categories
----------------------------------------------
_imagenet_kitchen.csv             35
_imagenet_dogs.csv                129
_imagenet_cats.csv                13
_imagenet_wearable.csv            56
_imagenet_landtransport.csv       45
_imagenet_birds.csv               60
----------------------------------------------
                                  338

References:
- github.com/don-tpanic/CSML_attention_project_pieces
"""

version_wnids = input('Version number (WNIDs): ')

import os
import csv
import numpy as np
import pandas as pd
from ..utils.paths import path_category_sets
from ..utils_cat_set_properties import *

filenames = [f for f in os.listdir(path_category_sets) if 'imagenet' in f]
category_sets = []

for filename in filenames:
    df = pd.read_csv(path_category_sets/filename)
    category_sets.append(list(df['wnid']))
    print(f'Found {df.shape[0]} classes in {filename}')

with open(path_category_sets/f'sem_v{version_wnids}_wnids.csv', 'w') as f:
    csv.writer(f).writerows(category_sets)
