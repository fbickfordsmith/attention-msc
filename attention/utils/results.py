"""
Define helper functions used for plotting.
"""

import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import squareform
from ..utils.category_sets import average_distance
from ..utils.metadata import df_baseline, representations, wnid2ind
from ..utils.paths import path_category_sets, path_results, path_training

def load_category_sets(type_category_set, version_wnids):
    f = open(
        path_category_sets/f'{type_category_set}_v{version_wnids}_wnids.csv')
    return [row for row in csv.reader(f, delimiter=',')]

def category_set_size(type_category_set, version_wnids):
    category_sets = load_category_sets(type_category_set, version_wnids)
    return [len(c) for c in category_sets]

def category_set_base_accuracy(type_category_set, version_wnids):
    category_sets = load_category_sets(type_category_set, version_wnids)
    stats = []
    for c in category_sets:
        inds_in = [wnid2ind[w] for w in c]
        inds_out = np.setdiff1d(range(1000), inds_in)
        stats.append([
            np.mean(df_baseline['accuracy'][inds_in]),
            np.mean(df_baseline['accuracy'][inds_out])])
    return pd.DataFrame(stats, columns=('acc_base_in', 'acc_base_out'))

def category_set_distance(type_category_set, version_wnids, measure='cosine'):
    category_sets = load_category_sets(type_category_set, version_wnids)
    if measure == 'cosine':
        distance = cosine_distances
    else:
        distance = euclidean_distances
    stats = []
    for c in category_sets:
        inds_in = [wnid2ind[w] for w in c]
        stats.append(average_distance(distance(representations[inds_in])))
    return pd.Series(stats, name=f'{measure}_mean')

def category_set_epochs(type_category_set, version_weights):
    return [
        len(pd.read_csv(path_training/filename, index_col=0))
        for filename in sorted(os.listdir(path_training))
        if f'{type_category_set}_v{version_weights}' in filename]

def category_set_summary(type_category_set, version_wnids, version_weights):
    df0 = category_set_base_accuracy(type_category_set, version_wnids)
    df1 = pd.read_csv(
        path_results/f'{type_category_set}_v{version_weights}_results.csv',
        index_col=0)
    return pd.DataFrame({
        'size': category_set_size(type_category_set, version_wnids),
        'similarity': 1 - category_set_distance(
            type_category_set, version_wnids, 'cosine'),
        'acc_base_in': df0['acc_base_in'],
        'acc_base_out': df0['acc_base_out'],
        'acc_trained_in': df1['acc_top1_in'],
        'acc_trained_out': df1['acc_top1_out'],
        'acc_change_in': df1['acc_top1_in'] - df0['acc_base_in'],
        'acc_change_out': df1['acc_top1_out'] - df0['acc_base_out'],
        'num_epochs': category_set_epochs(type_category_set, version_weights)})
