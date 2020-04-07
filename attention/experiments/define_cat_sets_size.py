"""
Define a set of 11 size-based category sets. These are subsets of ImageNet
categories that we choose to have varying size (number of categories) but approx
equal difficulty and visual similarity.

Method:
1. For 10,000 repeats
    a. Sample a candidate set of category_sets, {C_1', ..., C_11'}.
    b. Compute acc = [normalised_deviation_from_mean_acc(C_i) for i = 1:11].
    c. Compute dist = [normalised_deviation_from_mean_dist(C_i) for i = 1:11].
    d. If std(concat(acc, dist)) < current_lowest_std, keep category_set set.

Previous versions used
- category_set_sizes = [int(2**x) for x in range(9)]
- accdist = [score_acc(inds)+score_dist(inds) for inds in inds_sampled]
"""

version_wnids = input('Version number (WNIDs): ')

from ..utils_cat_set_properties import (average_distance, base_accuracy,
    score_acc, score_dist)
from ..utils.metadata import ind2wnid
from ..utils.paths import path_category_sets

category_set_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
accdist_bestscore = np.inf

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled = [
        np.random.choice(1000, size=s, replace=False) for s in category_set_sizes]
    accdist = [score_acc(inds) for inds in inds_sampled]
    accdist.extend([score_dist(inds) for inds in inds_sampled[1:]]) # first category_set always has distance of 0
    accdist_score = np.max(np.abs(accdist)) # similar results with accdist_score = np.std(accdist)
    if accdist_score < accdist_bestscore:
        inds_best = inds_sampled
        accdist_bestscore = accdist_score

print('Accuracy:', [round(base_accuracy(inds), 2) for inds in inds_best])
print('Distance:',
    [round(average_distance(distances(Z[inds])), 2) for inds in inds_best])

with open(path_category_sets/f'size_v{version_wnids}_wnids.csv', 'w') as f:
    for inds_category_set in inds_best:
        csv.writer(f).writerow([ind2wnid[i] for i in inds_category_set])
