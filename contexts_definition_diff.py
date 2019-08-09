'''
Group ImageNet classes into 25 'difficulty contexts'.

Method:
1. Sort classes by the baseline accuracy of VGG16.
2. Split into 20 disjoint sets of classes.
3. Sample 5 additional sets in order to get better coverage of context
    accuracies in the range [0.2, 0.4].
'''

import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/attention/'
path_baseline = f'{path}results/baseline_classwise_acc.csv'
path_synsets = f'{path}metadata/synsets.txt'

def baseline_acc_inds(inds):
    return np.mean(df_base['accuracy'][inds])

def baseline_acc_context(inds_incontext):
    inds_outofcontext = [np.setdiff1d(range(1000), inds_in) for inds_in in inds_incontext]
    df_contexts = pd.DataFrame()
    df_contexts['incontext_base'] = [baseline_acc_inds(inds_in) for inds_in in inds_incontext]
    df_contexts['outofcontext_base'] = [baseline_acc_inds(inds_out) for inds_out in inds_outofcontext]
    return df_contexts

def sample_below_threshold(threshold, context_size=50):
    return np.random.choice(
        df_base.loc[df_base['accuracy']<=threshold].index,
        size=context_size,
        replace=False)

def check_coverage(scores, interval_ends):
    intervals = [[interval_ends[i], interval_ends[i+1]] for i in range(len(interval_ends)-1)]
    return np.all([np.any((L < scores) & (scores < U)) for L, U in intervals])

wnids = open(path_synsets).read().splitlines()
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}

df_base = pd.read_csv(path_baseline, index_col=0)
df_base.sort_values(by='accuracy', ascending=True, inplace=True)
inds_split = np.array([list(inds) for inds in np.split(df_base.index, 20)])

thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]
interval_ends = [0.2, 0.25, 0.3, 0.35, 0.4]
intervals_covered = False
counter = 0

while not intervals_covered:
    inds_sampled = np.array([sample_below_threshold(t) for t in thresholds])
    acc_sampled = np.array([baseline_acc_inds(i) for i in inds_sampled])
    intervals_covered = check_coverage(acc_sampled, interval_ends)
    counter += 1
    if counter > 1000: break

if intervals_covered:
    inds_all = np.concatenate((inds_split, inds_sampled), axis=0)
    wnids_all = np.vectorize(ind2wnid.get)(inds_all)
    pd.DataFrame(wnids_all).to_csv(
        f'{path}contexts/diffcontexts_wnids.csv', header=False, index=False)
else:
    print('Suitable contexts not found')
