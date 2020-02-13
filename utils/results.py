"""
Define helper functions used for plotting.
"""

import os, sys
sys.path.append('..')

import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import squareform
from utils.paths import path_repo
from utils.metadata import *

def load_contexts(type_context, version_wnids):
    f = open(path_repo/f'data/contexts/{type_context}_v{version_wnids}_wnids.csv')
    return [row for row in csv.reader(f, delimiter=',')]

def context_size(type_context, version_wnids):
    contexts = load_contexts(type_context, version_wnids)
    return [len(c) for c in contexts]

def context_base_accuracy(type_context, version_wnids):
    contexts = load_contexts(type_context, version_wnids)
    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        inds_out = np.setdiff1d(range(1000), inds_in)
        stats.append([
            np.mean(df_baseline['accuracy'][inds_in]),
            np.mean(df_baseline['accuracy'][inds_out])])
    return pd.DataFrame(stats, columns=('acc_base_in', 'acc_base_out'))

def context_distance(type_context, version_wnids, measure='cosine'):
    contexts = load_contexts(type_context, version_wnids)
    if measure == 'cosine':
        distance = cosine_distances
    else:
        distance = euclidean_distances
    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        stats.append(average_distance(distance(Z[inds_in])))
    return pd.Series(stats, name=f'{measure}_mean')

def context_epochs(type_context, version_weights):
    return [
        len(pd.read_csv(path_repo/f'data/training/{filename}', index_col=0))
        for filename in sorted(os.listdir(path_repo/'data/training/'))
        if f'{type_context}_v{version_weights}' in filename]

def context_summary(type_context, version_wnids, version_weights):
    df0 = context_base_accuracy(type_context, version_wnids)
    df1 = pd.read_csv(
        path_repo/f'data/results/{type_context}_v{version_weights}_results.csv',
        index_col=0)
    return pd.DataFrame({
        'size': context_size(type_context, version_wnids),
        'similarity': 1 - context_distance(type_context, version_wnids, 'cosine'),
        'acc_base_in': df0['acc_base_in'],
        'acc_base_out': df0['acc_base_out'],
        'acc_trained_in': df1['acc_top1_in'],
        'acc_trained_out': df1['acc_top1_out'],
        'acc_change_in': df1['acc_top1_in'] - df0['acc_base_in'],
        'acc_change_out': df1['acc_top1_out'] - df0['acc_base_out'],
        'num_epochs': context_epochs(type_context, version_weights)})

def average_distance(Zdist):
    if Zdist.shape == (1, 1):
        return 0
    else:
        return np.mean(squareform(Zdist, checks=False))
