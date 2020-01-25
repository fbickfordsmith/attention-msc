'''
Define helper functions used for plotting.
'''

import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import squareform

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_base = f'{path}results/base_results.csv'
path_synsets = f'{path}metadata/synsets.txt'
path_representations = f'{path}representations/representations_mean.npy'

df_base = pd.read_csv(path_base, index_col=0)
mean_acc = np.mean(df_base['accuracy'])
std_acc = np.std(df_base['accuracy'])
lb_acc = mean_acc - std_acc
ub_acc = mean_acc + std_acc
inds_av_acc = np.flatnonzero((lb_acc<df_base['accuracy']) & (df_base['accuracy']<ub_acc))

wnids = open(path_synsets).read().splitlines()
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

Z = np.load(path_representations)

def load_contexts(type_context, version_wnids):
    f = open(f'{path}contexts/{type_context}_v{version_wnids}_wnids.csv')
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
            np.mean(df_base['accuracy'][inds_in]),
            np.mean(df_base['accuracy'][inds_out])])
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
        len(pd.read_csv(f'{path}training/{filename}', index_col=0))
        for filename in sorted(os.listdir(f'{path}training/'))
        if f'{type_context}_v{version_weights}' in filename]

def context_summary(type_context, version_wnids, version_weights):
    df0 = context_base_accuracy(type_context, version_wnids)
    df1 = pd.read_csv(f'{path}results/{type_context}_v{version_weights}_results.csv', index_col=0)
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
