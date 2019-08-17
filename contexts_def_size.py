'''
Select a set of 11 'size contexts', {C_1, ..., C_11}. These contexts are
non-disjoint subsets of ImageNet classes that we choose to have
- approximately equal difficulty and similarity
- varying size (|C| in [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256])

Method:
For 10,000 repeats
1. Sample a candidate set of contexts, {C_1', ..., C_11'}
2. Compute acc = [normalised_deviation_from_mean_acc(C_i) for i = 1:11]
3. Compute dist = [normalised_deviation_from_mean_dist(C_i) for i = 1:11]
4. If std(concat(acc, dist)) < current_lowest_std, keep context set
'''

from contexts_definition import *

type_context = 'size'
version_wnids = 'v' + input('Version number (WNIDs): ')

context_sizes = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256]
# context_sizes = [int(2**x) for x in range(9)]

min_std_accdist = np.inf

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_contexts = [
        np.random.choice(1000, size=s, replace=False) for s in context_sizes]
        # np.random.choice(inds_av_acc, size=s, replace=False) for s in context_sizes]
    accdist = [score_acc(inds) for inds in inds_contexts]
    accdist.extend([score_acc(inds) for inds in inds_contexts[1:]]) # first context always has distance of 0
    # accdist = [score_acc(inds)+score_dist(inds) for inds in inds_contexts]
    std_accdist = np.std(accdist)
    if std_accdist < min_std_accdist:
        inds_keep = inds_contexts
        min_std_accdist = std_accdist

print('Accuracy:', [round(base_accuracy(inds), 2) for inds in inds_keep])
print('Distance:', [round(average_distance(distances(Z[inds])), 2) for inds in inds_keep])

with open(f'{path}contexts/{type_context}_{version_wnids}_wnids.csv', 'w') as f:
    for context in inds_contexts:
        csv.writer(f).writerow([ind2wnid[i] for i in context])

# min_std_acc = np.inf
# inds_keep = None
#
# for i in range(10000):
#     if i % 1000 == 0:
#         print(f'i = {i:05}')
#     inds_contexts = [
#         np.random.choice(inds_av_acc, size=s, replace=False) for s in context_sizes]
#     acc = [score_acc(inds) for inds in inds_contexts]
#     std_acc = np.std(acc)
#     dist_in_bounds = np.all([check_dist_in_bounds(inds) for inds in inds_contexts[1:]]) # first context always has distance of 0
#     if dist_in_bounds and (std_acc_sampled < min_std_acc):
#         inds_keep = inds_contexts
#         min_std_acc = std_acc
#
# if inds_keep is not None:
#     print('Accuracy:', [round(base_accuracy(inds), 2) for inds in inds_keep])
#     print('Distance:', [round(average_distance(distances(Z[inds])), 2) for inds in inds_keep])
#     with open(f'{path}contexts/{type_context}_{version_wnids}_wnids.csv', 'w') as f:
#         for context in inds_contexts:
#             csv.writer(f).writerow([ind2wnid[i] for i in context])
# else:
#     print('Suitable contexts not found')
