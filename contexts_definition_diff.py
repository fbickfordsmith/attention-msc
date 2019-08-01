'''
Group ImageNet classes into 20 'difficulty contexts'.

Method:
1. Sort classes by the baseline accuracy of VGG16.
2. Split into 20 disjoint sets of classes.
'''

import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
df = pd.read_csv(f'{path}results/baseline_classwise_acc.csv', index_col=0)
df.sort_values(by='accuracy', ascending=False, inplace=True)
wnids_split = np.array([np.array(d['wnid']) for d in np.split(df, 20, axis=0)])
pd.DataFrame(wnids_split).to_csv(
    f'{path}contexts/diffcontexts_wnids_v1.csv', header=False, index=False)
