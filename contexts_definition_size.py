'''
Group ImageNet classes into 18 'size contexts'.

Method:
1. Find the ImageNet classes with baseline accuracy within 1 standard deviation
    of the average accuracy.
2. From these, sample contexts (sets of classes) of size in
    [int(1.4**i) for i in range(2, 20)]. These contexts can have common members.
3. Repeat sampling if the standard deviation of baseline accuracy across the
    contexts is greater than 2% (ensures contexts have approximately equal
    difficulty).
'''

import csv
import numpy as np
import pandas as pd

path = '/Users/fbickfordsmith/Google Drive/Project/'
path_baseline = f'{path}attention/results/baseline_classwise_acc.csv'
path_contexts = f'{path}attention/contexts/sizecontexts_wnids.csv'
path_synsets = f'{path}attention/metadata/synsets.txt'

df = pd.read_csv(path_baseline, index_col=0)
wnids = open(path_synsets).read().splitlines()
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

mean = np.mean(df['accuracy'])
std = np.std(df['accuracy'])
classes_within_1std = np.flatnonzero((df['accuracy']>mean-std) & (df['accuracy']<mean+std))
print(f'{len(classes_within_1std)} classes within 1 std of mean baseline accuracy')
context_sizes = [int(1.4**i) for i in range(2, 20)]

std_acc = 1
while std_acc > 0.02:
    contexts, accuracy = [], []
    for size in context_sizes:
        indices = np.random.choice(classes_within_1std, size=size, replace=False)
        wnids_c = list(df['wnid'][indices])
        contexts.append(wnids_c)
        accuracy.append(np.mean(df['accuracy'][[wnid2ind[w] for w in wnids_c]]))
    std_acc = np.std(np.array(accuracy))
    print(std_acc)

# with open(path+'attention/contexts/sizecontexts_wnids.csv', 'w') as f:
#     wr = csv.writer(f)
#     for c in contexts:
#         wr.writerow(c)
