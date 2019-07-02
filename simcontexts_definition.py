'''
Take the mean VGG16 representations of ImageNet classes.

For 5 different seed points in this representation space, define 4 non-disjoint
'contexts'

'''

# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
Xraw = np.load(path+'npy/mean_activations.npy')
Xsim = cosine_similarity(Xraw)
df = pd.read_csv(path+'csv/baseline_classwise.csv', index_col=0)

def sample_inds(ind_max, ind_exclude, size=49):
    options = np.setdiff1d(np.arange(ind_max), ind_exclude)
    return np.random.choice(options, size=size, replace=False)

seed_inds = np.random.choice(np.arange(1000), size=5, replace=False)
batches, similarities = [], []

for i in seed_inds:
    sorted_inds = np.argsort(Xsim[i])[-2::-1] # -2 => don't include seed index

    for ind_max in np.linspace(50, 999, 4, dtype=int):
        sampled_inds = sorted_inds[sample_inds(ind_max, i)]
        batch_inds = np.insert(sampled_inds, 0, i)
        batches.append(batch_inds)
        similarities.append(np.mean(cosine_similarity(Xraw[batch_inds])))

sets_df = pd.DataFrame()
sets_df['wnids'] = [list(df.iloc[b]['wnid']) for b in batches]
sets_df['num_examples'] = [np.sum(df.iloc[b]['num_examples']) for b in batches]
sets_df['num_correct'] = [np.sum(df.iloc[b]['num_correct']) for b in batches]
sets_df['incontext_acc'] = sets_df['num_correct'] / sets_df['num_examples']

outofcontext_acc = []

for i in range(20):
    ind_not_i = [j for j in range(20) if j != i]
    outofcontext_acc.append(
        np.sum(sets_df['num_correct'][ind_not_i]) /
        np.sum(sets_df['num_examples'][ind_not_i]))

sets_df['outofcontext_acc'] = outofcontext_acc
sets_df['incontext_sim'] = similarities

sets_df = sets_df.astype({
    'wnids':object,
    'num_examples':int,
    'num_correct':int,
    'incontext_acc':float,
    'outofcontext_acc':float,
    'incontext_sim':float})

sets_df.to_csv(path+'csv/simcontexts_definition.csv')
simcontext_wnids = np.array(wnids)
pd.DataFrame(simcontext_wnids).to_csv(
    path+'csv/simcontexts_wnids.csv', header=False, index=False)
