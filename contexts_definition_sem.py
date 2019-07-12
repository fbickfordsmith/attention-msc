'''
Group ImageNet classes into 40 'semantic contexts'.

Method:
1. Take the semantic groupings defined by Ken.
2. Ignore the cats (felidae) group because it only contains 15 classes.
3. For all other groups, randomly sample 35 classes to form a context.

References:
- github.com/don-tpanic/CSML_attention_project_pieces
'''

import os
import csv
import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/attention/contexts/'
filenames = [f for f in os.listdir(path) if 'imagenet' in f]
wnids, num_classes = [], []

for f in filenames:
    df = pd.read_csv(path+f)
    wnids.append(list(df['wnid']))
    num_classes.append(df.shape[0])
    print(f'Found {df.shape[0]} classes in {f}')

with open(path+'semcontexts_wnids.csv', 'w') as f:
    csv.writer(f).writerows(wnids)
