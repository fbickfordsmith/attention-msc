"""
Define helper functions used in `contexts_def_[type_context].py` for
[type_context] in {diff, sem, sim, size}.
"""

import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import squareform
from utils.metadata import *

Zdist = cosine_distances(Z)
mean_dist = np.mean(squareform(Zdist, checks=False))
std_dist = np.std(squareform(Zdist, checks=False))
lb_dist = mean_dist - std_dist
ub_dist = mean_dist + std_dist

def average_distance(Zdist):
    if Zdist.shape == (1, 1):
        return 0
    else:
        return np.mean(squareform(Zdist, checks=False))

def base_accuracy(inds):
    return np.mean(df_baseline['accuracy'][inds])

def score_acc(inds):
    return (base_accuracy(inds) - mean_acc) / std_acc

def score_dist(inds):
    return (average_distance(cosine_distances(Z[inds])) - mean_dist) / std_dist

def check_coverage(scores, interval_ends):
    intervals = [
        [interval_ends[i], interval_ends[i+1]]
        for i in range(len(interval_ends)-1)]
    return np.all([np.any((L < scores) & (scores < U)) for L, U in intervals])

def check_dist_in_bounds(inds):
    d = average_distance(cosine_distances(Z[inds]))
    return (lb_dist<d) & (d<ub_dist)

def sample_below_acc(acc, context_size=50):
    return np.random.choice(
        df_baseline.loc[df_baseline['accuracy']<=acc].index,
        size=context_size,
        replace=False)
