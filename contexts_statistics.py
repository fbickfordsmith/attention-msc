import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from scipy.spatial.distance import squareform

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_baseline = f'{path}results/baseline_classwise_acc.csv'
path_synsets = f'{path}metadata/synsets.txt'
path_activations = f'{path}activations/activations_mean.npy'

df_base = pd.read_csv(path_baseline, index_col=0)
mean_acc = np.mean(df_base['accuracy'])
std_acc = np.std(df_base['accuracy'])
lb_acc = mean_acc - std_acc
ub_acc = mean_acc + std_acc
inds_av_acc = np.flatnonzero((lb_acc<df_base['accuracy']) & (df_base['accuracy']<ub_acc))

wnids = open(path_synsets).read().splitlines()
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

X = np.load(path_activations)

def context_size(type_context):
    with open(f'{path}contexts/{type_context}contexts_wnids.csv') as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]
    return [len(c) for c in contexts]

def context_baseline_accuracy(type_context):
    with open(f'{path}contexts/{type_context}contexts_wnids.csv') as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]
    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        inds_out = list(set(range(1000)) - set(inds_in))
        stats.append([
            np.mean(df_base['accuracy'][inds_in]),
            np.mean(df_base['accuracy'][inds_out])])
    return pd.DataFrame(stats, columns=('incontext_base', 'outofcontext_base'))

def context_distance(type_context, measure='cosine'):
    with open(f'{path}contexts/{type_context}contexts_wnids.csv') as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]
    if measure == 'cosine':
        distance = cosine_distances
    else:
        distance = euclidean_distances
    stats = []
    for c in contexts:
        inds_in = [wnid2ind[w] for w in c]
        stats.append(average_distance(distance(X[inds_in])))
    return pd.Series(stats, name=f'mean_{measure}_distance')

def context_epochs(type_context, num_contexts):
    return [
        len(pd.read_csv(f'{path}training/{type_context}context{i:02}_training.csv', index_col=0))
        for i in range(num_contexts)]

def context_summary(type_context):
    df0 = context_baseline_accuracy(type_context)
    df1 = pd.read_csv(f'{path}results/{type_context}contexts_trained_metrics.csv', index_col=0)
    df_sum = pd.DataFrame()
    df_sum['size'] = context_size(type_context)
    df_sum['similarity'] = 1 - context_distance(type_context, 'cosine')
    df_sum['incontext_base'] = df0['incontext_base']
    df_sum['outofcontext_base'] = df0['outofcontext_base']
    df_sum['incontext_trained'] = df1['incontext_acc_top1']
    df_sum['outofcontext_trained'] = df1['outofcontext_acc_top1']
    df_sum['incontext_change'] = df_sum['incontext_trained'] - df_sum['incontext_base']
    df_sum['outofcontext_change'] = df_sum['outofcontext_trained'] - df_sum['outofcontext_base']
    df_sum['num_epochs'] = context_epochs(type_context, len(df1))
    return df_sum

def average_distance(Xdist):
    if Xdist.shape == (1, 1):
        return 0
    else:
        return np.mean(squareform(Xdist, checks=False))
