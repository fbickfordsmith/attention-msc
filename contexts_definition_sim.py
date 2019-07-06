'''
Group ImageNet classes into 40 'similarity contexts'.

Method:
1. Take the mean VGG16 representations (4096-dim vectors) of ImageNet classes.
2. For 10 different seed points, sample 4 sets of points, each of which is
    defined as a context. These sets can have overlapping members.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
Xraw = np.load(f'{path}npy/mean_activations.npy')
Xdist = pairwise_distances(Xraw)
df = pd.read_csv(f'{path}results/baseline_classwise_acc.csv', index_col=0)

def sample_inds(ind_max, ind_exclude, size=49):
    options = np.setdiff1d(np.arange(ind_max), ind_exclude)
    return np.random.choice(options, size=size, replace=False)

num_seeds = 10
seed_inds = np.random.choice(np.arange(1000), size=num_seeds, replace=False)
batches, distances, stddevs = [], [], []
interval_ends = np.linspace(50, 999, 4, dtype=int)

for i in seed_inds:
    sorted_inds = np.argsort(Xdist[i])[1:] # 1 => don't include seed index
    for ind_end in interval_ends:
        sampled_inds = sorted_inds[sample_inds(ind_end, i)]
        batch_inds = np.insert(sampled_inds, 0, i)
        batches.append(batch_inds)
        distances.append(np.mean(pairwise_distances(Xraw[batch_inds])))
        stddevs.append(np.std(pairwise_distances(Xraw[batch_inds])))

sets_df = pd.DataFrame()
sets_df['wnids'] = [list(df.iloc[b]['wnid']) for b in batches]
sets_df['num_examples'] = [np.sum(df.iloc[b]['num_examples']) for b in batches]
sets_df['num_correct'] = [np.sum(df.iloc[b]['num_correct']) for b in batches]
sets_df['incontext_acc'] = sets_df['num_correct'] / sets_df['num_examples']

outofcontext_acc = []
for i in range(len(distances)):
    ind_not_i = [j for j in range(len(distances)) if j != i]
    outofcontext_acc.append(
        np.sum(sets_df['num_correct'][ind_not_i]) /
        np.sum(sets_df['num_examples'][ind_not_i]))

sets_df['outofcontext_acc'] = outofcontext_acc
sets_df['incontext_meandistance'] = distances
sets_df['incontext_stddistance'] = stddevs

sets_df = sets_df.astype({
    'wnids':object,
    'num_examples':int,
    'num_correct':int,
    'incontext_acc':float,
    'outofcontext_acc':float,
    'incontext_meandistance':float,
    'incontext_stddistance':float})

sets_df.to_csv(f'{path}contexts/simcontexts_stats.csv')
simcontext_wnids = np.array([list(df.iloc[b]['wnid']) for b in batches])
pd.DataFrame(simcontext_wnids).to_csv(
    f'{path}contexts/simcontexts_wnids.csv', header=False, index=False)
