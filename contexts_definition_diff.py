'''
Group ImageNet classes into 20 'difficulty contexts'.

Method:
1. Sort classes by the baseline accuracy of VGG16.
2. Split into 20 disjoint sets of classes.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
df = pd.read_csv(f'{path}results/baseline_classwise_acc.csv', index_col=0)
sorted_df = df.sort_values(by='accuracy', ascending=False)
sets_df = pd.DataFrame()

sorted_wnids = np.array(sorted_df['wnid'], dtype=str)
sorted_num_examples = np.array(sorted_df['num_examples'])
sorted_num_correct = np.array(sorted_df['num_correct'])

sets_df['wnids'] = [list(arr) for arr in np.split(sorted_wnids, 20)]
sets_df['num_examples'] = [np.sum(arr) for arr in np.split(sorted_num_examples, 20)]
sets_df['num_correct'] = [np.sum(arr) for arr in np.split(sorted_num_correct, 20)]
sets_df['incontext_acc'] = sets_df['num_correct'] / sets_df['num_examples']

outofcontext_acc = []
for i in range(20):
    ind_not_i = [j for j in range(20) if j != i]
    outofcontext_acc.append(
        np.sum(sets_df['num_correct'][ind_not_i]) /
        np.sum(sets_df['num_examples'][ind_not_i]))
sets_df['outofcontext_acc'] = outofcontext_acc

sets_df = sets_df.astype({
    'wnids':object,
    'num_examples':int,
    'num_correct':int,
    'incontext_acc':float,
    'outofcontext_acc':float})

sets_df.to_csv(f'{path}contexts/diffcontexts_stats.csv')
diffcontext_wnids = np.array(np.split(sorted_wnids, 20))
pd.DataFrame(diffcontext_wnids).to_csv(
    f'{path}contexts/diffcontexts_wnids.csv', header=False, index=False)
