'''
Group ImageNet classes into 6 'semantic contexts'.

$ python3 contexts_definition_sem.py
Found 35 classes in imagenet_kitchen.csv
Found 129 classes in imagenet_dogs.csv
Found 13 classes in imagenet_cats.csv
Found 56 classes in imagenet_wearable.csv
Found 45 classes in imagenet_landtransport.csv
Found 60 classes in imagenet_birds.csv

Total number of classes: 338.

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
