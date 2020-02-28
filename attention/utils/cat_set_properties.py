"""
Define helper functions used in `define_cat_sets_[type_category_set].py` for
[type_category_set] in {diff, sem, sim, size}.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from ..utils.metadata import (
    df_baseline, mean_acc, mean_dist, std_acc, std_dist, representations)

def average_distance(distance_matrix):
    if distance_matrix.shape == (1, 1):
        return 0
    else:
        return np.mean(squareform(distance_matrix, checks=False))

def base_accuracy(inds):
    return np.mean(df_baseline['accuracy'][inds])

def check_coverage(scores, interval_ends):
    intervals = [
        [interval_ends[i], interval_ends[i+1]]
        for i in range(len(interval_ends) - 1)]
    return np.all([np.any((l < scores) & (scores < u)) for l, u in intervals])

def check_dist_in_bounds(inds):
    d = average_distance(cosine_distances(representations[inds]))
    return ((mean_dist - std_dist) < d) & (d < (mean_dist + std_dist))

def sample_below_acc(acc, category_set_size=50):
    return np.random.choice(
        df_baseline.loc[df_baseline['accuracy'] <= acc].index,
        size=category_set_size,
        replace=False)

def score_acc(inds):
    return (base_accuracy(inds) - mean_acc) / std_acc

def score_dist(inds):
    return (
        (average_distance(cosine_distances(representations[inds])) - mean_dist)
        / std_dist)
