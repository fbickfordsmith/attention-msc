"""
Define a set of 11 'size contexts'. These are subsets of ImageNet classes
that we choose to have varying size (number of classes) but approx equal
difficulty and visual similarity.

Method:
1. For 10,000 repeats
    a. Sample a candidate set of contexts, {C_1', ..., C_11'}.
    b. Compute acc = [normalised_deviation_from_mean_acc(C_i) for i = 1:11].
    c. Compute dist = [normalised_deviation_from_mean_dist(C_i) for i = 1:11].
    d. If std(concat(acc, dist)) < current_lowest_std, keep context set.

Previous versions used
- context_sizes = [int(2**x) for x in range(9)]
- random.choice(inds_av_acc) instead of random.choice(1000)
- accdist = [score_acc(inds)+score_dist(inds) for inds in inds_sampled]
"""

version_wnids = input('Version number (WNIDs): ')

import sys
sys.path.append('..')

from utils.paths import path_repo
from utils.contexts_definition import *

path_save = path_repo/f'data/contexts/size_v{version_wnids}_wnids.csv'
context_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]

accdist_bestscore = np.inf

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled = [
        np.random.choice(1000, size=s, replace=False) for s in context_sizes]
    accdist = [score_acc(inds) for inds in inds_sampled]
    accdist.extend([score_dist(inds) for inds in inds_sampled[1:]]) # first context always has distance of 0
    accdist_score = np.max(np.abs(accdist)) # similar results with accdist_score = np.std(accdist)
    if accdist_score < accdist_bestscore:
        inds_best = inds_sampled
        accdist_bestscore = accdist_score

print('Accuracy:', [round(base_accuracy(inds), 2) for inds in inds_best])
print('Distance:',
    [round(average_distance(distances(Z[inds])), 2) for inds in inds_best])

with open(path_save, 'w') as f:
    for inds_context in inds_best:
        csv.writer(f).writerow([ind2wnid[i] for i in inds_context])
