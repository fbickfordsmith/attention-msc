'''
We varying three properties of contexts: size, difficulty and similarity.

In our experiments we aim to vary only one property at a time, minimising
variation in the other properties. The way we defined the contexts for each
experiment was dictated by this desideratum:

- Size: vary size; approx-fix difficulty; approx-fix similarity
- Difficulty: exact-fix size; vary difficulty; approx-fix similarity
- Similarity: exact-fix size; approx-fix difficulty; vary similarity
'''

import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.spatial.distance import squareform

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_base = f'{path}results/base_results.csv'
path_synsets = f'{path}metadata/synsets.txt'
path_representations = f'{path}representations/representations_mean.npy'

def average_distance(Zdist):
    if Zdist.shape == (1, 1):
        return 0
    else:
        return np.mean(squareform(Zdist, checks=False))

def base_accuracy(inds):
    return np.mean(df_base['accuracy'][inds])

def score_acc(inds):
    return (base_accuracy(inds) - mean_acc) / std_acc

def score_dist(inds):
    return (average_distance(distances(Z[inds])) - mean_dist) / std_dist

def check_coverage(scores, interval_ends):
    intervals = [[interval_ends[i], interval_ends[i+1]] for i in range(len(interval_ends)-1)]
    return np.all([np.any((L < scores) & (scores < U)) for L, U in intervals])

def check_dist_in_bounds(inds):
    d = average_distance(distances(Z[inds]))
    return (lb_dist<d) & (d<ub_dist)

def sample_below_acc(acc, context_size=50):
    return np.random.choice(
        df_base.loc[df_base['accuracy']<=acc].index, size=context_size, replace=False)

wnids = open(path_synsets).read().splitlines()
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}

df_base = pd.read_csv(path_base, index_col=0)
mean_acc = np.mean(df_base['accuracy'])
std_acc = np.std(df_base['accuracy'])
lb_acc = mean_acc - std_acc
ub_acc = mean_acc + std_acc
inds_av_acc = np.flatnonzero((lb_acc<df_base['accuracy']) & (df_base['accuracy']<ub_acc))

Z = np.load(path_representations)
distances = cosine_distances # distances = euclidean_distances
Zdist = distances(Z)
mean_dist = np.mean(squareform(Zdist, checks=False))
std_dist = np.std(squareform(Zdist, checks=False))
lb_dist = mean_dist - std_dist
ub_dist = mean_dist + std_dist
