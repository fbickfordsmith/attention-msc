"""
Define a set of 20 difficulty-based category sets. These are subsets of ImageNet
categories that we choose to have varying difficulty (average error rate of
VGG16) but equal size and approx equal visual similarity.

Method:
1. Sort categories by the base accuracy of VGG16.
2. Split into 20 disjoint sets of categories.
3. Sample 5 additional sets in order to get better coverage of category set
    accuracies in the range [0.2, 0.4].
"""

version_wnids = input('Version number (WNIDs): ')

from ..utils_cat_set_properties import (
    average_distance, base_accuracy, check_coverage, score_dist)
from ..utils.metadata import ind2wnid
from ..utils.paths import path_category_sets

df_baseline.sort_values(by='accuracy', ascending=True, inplace=True)
inds_split = np.array([list(inds) for inds in np.split(df_baseline.index, 20)])
thresholds = [0.35, 0.4, 0.45, 0.5, 0.55]
interval_ends = [0.2, 0.25, 0.3, 0.35, 0.4]
intervals_covered = False
dist_bestscore = np.inf
inds_best = None

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled = np.array([sample_below_acc(t) for t in thresholds])
    acc = [base_accuracy(inds) for inds in inds_sampled]
    dist = [score_dist(inds) for inds in inds_sampled]
    intervals_covered = check_coverage(np.array(acc), interval_ends)
    dist_score = np.max(np.abs(dist)) # similar results with dist_score = np.std(dist)
    if intervals_covered and (dist_score < dist_bestscore):
        inds_best = inds_sampled
        dist_bestscore = dist_score

if inds_best is not None:
    print('Accuracy:', [round(base_accuracy(inds), 2) for inds in inds_best])
    print('Distance:',
        [round(average_distance(distances(Z[inds])), 2) for inds in inds_best])
    inds_all = np.concatenate((inds_split, inds_best), axis=0)
    wnids_all = np.vectorize(ind2wnid.get)(inds_all)
    pd.DataFrame(wnids_all).to_csv(
        path_category_sets/f'diff_v{version_wnids}_wnids.csv',
        header=False,
        index=False)
else:
    print('Suitable category sets not found')
