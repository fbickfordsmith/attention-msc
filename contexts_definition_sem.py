'''
Group ImageNet classes into 6 'semantic contexts'.

References:
- github.com/don-tpanic/CSML_attention_project_pieces
'''

import os
import csv
import numpy as np
import pandas as pd

path_contexts = '/home/freddie/attention/contexts/'
# path_contexts = '/Users/fbickfordsmith/Google Drive/Project/attention/contexts/'
filenames = [f for f in os.listdir(path_contexts) if 'imagenet' in f]
wnids = []

for filename in filenames:
    df = pd.read_csv(path_contexts+filename)
    wnids.append(list(df['wnid']))
    print(f'Found {df.shape[0]} classes in {filename}')

with open(f'{path_contexts}semcontexts_wnids.csv', 'w') as f:
    csv.writer(f).writerows(wnids)
