'''
Define a set of 6 'semantic contexts'. These are subsets of ImageNet classes
that are conceptually similar, as judged by humans.

Filename                     Number of classes
_imagenet_kitchen.csv        35
_imagenet_dogs.csv           129
_imagenet_cats.csv           13
_imagenet_wearable.csv       56
_imagenet_landtransport.csv  45
_imagenet_birds.csv          60

Total: 338 classes.

References:
- github.com/don-tpanic/CSML_attention_project_pieces
'''

import os
import csv
import numpy as np
import pandas as pd

path_contexts = '/Users/fbickfordsmith/Google Drive/Project/attention/contexts/'
filenames = [f for f in os.listdir(path_contexts) if 'imagenet' in f]
contexts = []

for filename in filenames:
    df = pd.read_csv(path_contexts+filename)
    contexts.append(list(df['wnid']))
    print(f'Found {df.shape[0]} classes in {filename}')

with open(f'{path_contexts}semcontexts_wnids.csv', 'w') as f:
    csv.writer(f).writerows(contexts)
