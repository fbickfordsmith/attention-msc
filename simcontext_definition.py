'''
Take the mean VGG16 representations of ImageNet classes.

For 5 different seed points in this representation space, define 4 non-disjoint
sets of...

'''

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
Xraw = np.load(path+'npy/mean_activations.npy')
Xsim = cosine_similarity(Xraw)
df = pd.read_csv(path+'csv/baseline_classwise.csv', index_col=0)
ind2name = {ind:name for ind, name in enumerate(df['name'])}

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
        batches.append([ind2name[j] for j in batch_inds])
        similarities.append(np.mean(cosine_similarity(Xraw[batch_inds])))

pd.DataFrame(batches).to_csv('csv/simcontexts_wnids.csv')
# batches = np.array(batches)
# similarities = np.array(similarities)
