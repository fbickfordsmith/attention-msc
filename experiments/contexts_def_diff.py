'''
Define a set of 20 'difficulty contexts'. These are subsets of ImageNet classes
that we choose to have varying difficulty (average error rate of VGG16) but
equal size and approx equal visual similarity.

Method:
1. Sort classes by the base accuracy of VGG16.
2. Split into 20 disjoint sets of classes.
3. Sample 5 additional sets in order to get better coverage of context
    accuracies in the range [0.2, 0.4].
'''

from contexts_definition import *

type_context = 'diff'
version_wnids = 'v' + input('Version number (WNIDs): ')

df_base.sort_values(by='accuracy', ascending=True, inplace=True)
inds_split = np.array([list(inds) for inds in np.split(df_base.index, 20)])
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
    print('Distance:', [round(average_distance(distances(Z[inds])), 2) for inds in inds_best])
    inds_all = np.concatenate((inds_split, inds_best), axis=0)
    wnids_all = np.vectorize(ind2wnid.get)(inds_all)
    pd.DataFrame(wnids_all).to_csv(
        f'{path}contexts/{type_context}_{version_wnids}_wnids.csv', header=False, index=False)
else:
    print('Suitable contexts not found')
