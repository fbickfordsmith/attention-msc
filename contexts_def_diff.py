'''
Group ImageNet classes into 25 'difficulty contexts'.

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
dist_in_bounds = False
counter = 0

while not intervals_covered or not dist_in_bounds:
    inds_sampled = np.array([sample_below_acc(t) for t in thresholds])
    acc_sampled = np.array([base_accuracy(i) for i in inds_sampled])
    intervals_covered = check_coverage(acc_sampled, interval_ends)
    dist_in_bounds = np.all([check_dist_in_bounds(inds) for inds in inds_sampled])
    counter += 1
    if counter > 1000: break
inds_keep = inds_sampled

# intervals_covered = False
# min_std_dist = np.inf
# inds_keep = None

# for _ in range(10000):
#     inds_sampled = np.array([sample_below_acc(t) for t in thresholds])
#     acc_sampled = np.array([base_accuracy(i) for i in inds_sampled])
#     intervals_covered = check_coverage(acc_sampled, interval_ends)
#     dists = [score_dist(inds) for inds in inds_sampled]
#     std_dist_sampled = np.std(dists)
#     if std_dist_sampled < min_std_dist:
#         inds_keep = inds_sampled
#         min_std_dist = std_dist_sampled

if intervals_covered and dist_in_bounds:
    print(counter)
    inds_all = np.concatenate((inds_split, inds_keep), axis=0)
    wnids_all = np.vectorize(ind2wnid.get)(inds_all)
    pd.DataFrame(wnids_all).to_csv(
        f'{path}contexts/{type_context}_{version_wnids}_wnids.csv', header=False, index=False)
else:
    print('Suitable contexts not found')
