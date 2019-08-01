import csv
import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_baseline = f'{path}results/baseline_classwise_acc.csv'
path_synsets = f'{path}metadata/synsets.txt'

df_baseline = pd.read_csv(path_baseline, index_col=0)
wnids = [line.rstrip('\n') for line in open(path_synsets)]
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

def context_baseline_accuracy(type_context):
    path_contexts = f'{path}contexts/{type_context}contexts_wnids.csv'
    with open(path_contexts) as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]

    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        inds_out = list(set(range(1000)) - set(inds_in))
        stats.append([
            np.mean(df_baseline['accuracy'][inds_in]),
            np.mean(df_baseline['accuracy'][inds_out])])

    return pd.DataFrame(
        stats, columns=('incontext_base', 'outofcontext_base'))

# def context_distance(type_context, measure='euclidean'):
#     if measure == 'euclidean':
#         ...
#
#     return
