'''
[OLD VERSION]
Group ImageNet classes into 10 'size contexts'.

Method:
1. Take the mean VGG16 representations (4096-dim vectors) of ImageNet classes.
2. For 10 different seed points, sample 4 sets of points, each of which is
    defined as a context. These sets can have overlapping members.
'''

import csv
import numpy as np
import pandas as pd

df = pd.read_csv('results/baseline_classwise_acc.csv', index_col=0)
mean = np.mean(df['accuracy'])
std = np.std(df['accuracy'])
classes_within_1std = np.flatnonzero(
    (df['accuracy']>mean-std) & (df['accuracy']<mean+std))
print(f'{len(classes_within_1std)} classes within 1 std of mean baseline accuracy')
context_sizes = [2**i for i in range(10)]
wnids = []
for size in context_sizes:
    indices = np.random.choice(classes_within_1std, size=size, replace=False)
    wnids.append(list(df['wnid'][indices]))
print('Context sizes =', [len(w) for w in wnids])
with open('contexts/sizecontexts_wnids.csv', 'w') as f:
    wr = csv.writer(f)
    for item in wnids:
        wr.writerow(item)
