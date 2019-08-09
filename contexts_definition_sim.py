import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
# from sklearn.metrics.pairwise import cosine_distances

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_baseline = f'{path}results/baseline_classwise_acc.csv'
path_synsets = f'{path}metadata/synsets.txt'

def check_coverage(scores, interval_ends):
    intervals = [[interval_ends[i], interval_ends[i+1]] for i in range(len(interval_ends)-1)]
    return np.all([np.any((L < scores) & (scores < U)) for L, U in intervals])

wnids = open(path_synsets).read().splitlines()
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}

X = np.load(f'{path}activations/activations_mean.npy')
distance = euclidean_distances
Xdist = distance(X)

num_seeds = 5
context_size = 50
inds_end = np.linspace(50, 999, 4, dtype=int)

interval_ends = np.arange(25, 85, 5)
intervals_covered = False
counter = 0

while not intervals_covered:
    inds_contexts, distances = [], []
    inds_seed = np.random.choice(1000, size=num_seeds, replace=False)
    for ind_seed in inds_seed:
        inds_sorted = np.argsort(Xdist[ind_seed])[1:] # 1 => don't include seed index
        for ind_end in inds_end:
            inds_sampled = np.random.choice(
                inds_sorted[:ind_end], size=context_size-1, replace=False)
            inds_sampled = np.insert(inds_sampled, 0, ind_seed)
            inds_contexts.append(inds_sampled)
            distances.append(np.mean(distance(X[inds_sampled])))
    intervals_covered = check_coverage(np.array(distances), interval_ends)
    counter += 1
    if counter > 1000: break

if intervals_covered:
    wnids_contexts = np.vectorize(ind2wnid.get)(np.array(inds_contexts))
    pd.DataFrame(wnids_contexts).to_csv(
        f'{path}contexts/simcontexts_wnids.csv', header=False, index=False)
else:
    print('Suitable contexts not found')
