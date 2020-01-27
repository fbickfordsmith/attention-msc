"""
Define a set of 6 'semantic contexts'. These are subsets of ImageNet classes
that are conceptually similar, as judged by humans.

File in `data/contexts/`     Number of classes
----------------------------------------------
_imagenet_kitchen.csv        35
_imagenet_dogs.csv           129
_imagenet_cats.csv           13
_imagenet_wearable.csv       56
_imagenet_landtransport.csv  45
_imagenet_birds.csv          60
----------------------------------------------
                             338

References:
- github.com/don-tpanic/CSML_attention_project_pieces
"""

version_wnids = input('Version number (WNIDs): ')

import os, sys
sys.path.append('..')

import csv
import numpy as np
import pandas as pd
from utils.paths import path_repo
from utils.contexts_definition import *

path_contexts = path_repo/'data/contexts/'
path_save = path_contexts/f'sem_v{version_wnids}_wnids.csv'

filenames = [f for f in os.listdir(path_contexts) if 'imagenet' in f]
contexts = []

for filename in filenames:
    df = pd.read_csv(path_contexts/filename)
    contexts.append(list(df['wnid']))
    print(f'Found {df.shape[0]} classes in {filename}')

with open(path_save, 'w') as f:
    csv.writer(f).writerows(contexts)
