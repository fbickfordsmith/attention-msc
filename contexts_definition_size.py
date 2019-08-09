'''
Group ImageNet classes into 18 'size contexts'.

Method:
1. Find the ImageNet classes with baseline accuracy within 1 standard deviation
    of the average accuracy.
2. From these, sample contexts (sets of classes) of size in [2^x for x = 0:9].
    These contexts can have common members.
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

wnids = open(path_synsets).read().splitlines()
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

df = pd.read_csv(path_baseline, index_col=0)
mean = np.mean(df['accuracy'])
std = np.std(df['accuracy'])
classes_within_1std = np.flatnonzero((mean-std<df['accuracy']) & (df['accuracy']<mean+std))
print(f'{len(classes_within_1std)} classes within 1 std of mean baseline accuracy')

context_sizes = [int(2**x) for x in range(9)] # 1, 2, 4, 8, 16, 32, 64, 128, 256
# context_sizes = [int(1.4**x) for x in range(2, 20)] # 1, 2, 3, 5, 7, 10, 14, 20, 28, 40, 56, 79, 111, 155, 217, 304, 426, 597

count = 0
score = 1
while score > 0.01:
    contexts, accuracy = [], []
    for size in context_sizes:
        indices = np.random.choice(classes_within_1std, size=size, replace=False)
        wnids_c = list(df['wnid'][indices])
        contexts.append(wnids_c)
        accuracy.append(np.mean(df['accuracy'][[wnid2ind[w] for w in wnids_c]]))
    score = np.std(np.array(accuracy))
    count += 1
    if count > 10000: break

if score <= 0.01:
    with open(path+'attention/contexts/sizecontexts_wnids.csv', 'w') as f:
        for c in contexts:
            csv.writer(f).writerow(c)
else:
    print('Suitable contexts not found')
