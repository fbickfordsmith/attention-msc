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
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/attention/ken-groupings/'
filenames = [
    'canidae_Imagenet.csv',
    'kitchen_Imagenet.csv',
    'cloth_Imagenet.csv',
    'ave_Imagenet.csv',
    # 'felidae_Imagenet.csv', # only 15 classes
    'land_trans_Imagenet.csv']

wnids = []
for filename in filenames:
    df = pd.read_csv(f'{path}filename')
    inds = np.random.choice(np.arange(df.shape[0]), size=35, replace=False)
    wnids.append(list(df['wnid'][inds]))

pd.DataFrame(np.array(wnids)).to_csv(
    f'{path}semcontexts_wnids.csv', header=False, index=False)
