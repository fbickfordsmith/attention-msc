import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_baseline = f'{path}results/baseline_classwise_acc.csv'
path_synsets = f'{path}metadata/synsets.txt'
path_activations = f'{path}activations/activations_mean.npy'

df_baseline = pd.read_csv(path_baseline, index_col=0)
wnids = open(path_synsets).read().splitlines()
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}
X = np.load(path_activations)

def context_baseline_accuracy(type_context):
    with open(f'{path}contexts/{type_context}contexts_wnids.csv') as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]
    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        inds_out = list(set(range(1000)) - set(inds_in))
        stats.append([
            np.mean(df_baseline['accuracy'][inds_in]),
            np.mean(df_baseline['accuracy'][inds_out])])
    return pd.DataFrame(stats, columns=('incontext_base', 'outofcontext_base'))

def context_distance(type_context, measure='euclidean'):
    with open(f'{path}contexts/{type_context}contexts_wnids.csv') as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]
    if measure == 'euclidean':
        distance = euclidean_distances
    else:
        distance = cosine_distances
    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        stats.append(np.mean(distance(X[inds_in])))
    return pd.DataFrame(stats, columns=(f'mean_{measure}_distance',))
