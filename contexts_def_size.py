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

from contexts_definition import *

context_sizes = [int(2**x) for x in range(9)] # 1, 2, 4, 8, 16, 32, 64, 128, 256

min_std_accdist = np.inf
inds_keep = None
for _ in range(10000):
    inds_contexts = [
        np.random.choice(inds_av_acc, size=context_size, replace=False)
        for context_size in context_sizes]
    accdist = [acc_score(inds)+dist_score(inds) for inds in inds_contexts]
    std_accdist = np.std(accdist)
    if std_accdist < min_std_accdist:
        inds_keep = inds_contexts
        min_std_accdist = std_accdist

# min_std_acc = np.inf
# inds_keep = None
#
# for _ in range(10000):
#     inds_contexts = [
#         np.random.choice(inds_av_acc, size=context_size, replace=False)
#         for context_size in context_sizes]
#     acc = [acc_score(inds) for inds in inds_contexts]
#     std_acc_sampled = np.std(acc)
#     dist_in_bounds = np.all([check_dist_in_bounds(inds) for inds in inds_contexts])
#     if dist_in_bounds and (std_acc_sampled < min_std_acc):
#         inds_keep = inds_contexts
#         min_std_acc = std_acc_sampled

if inds_keep is not None:
    print(min_std_accdist)
    # with open(f'{path}contexts/sizecontexts_wnids.csv', 'w') as f:
    with open(f'{path}contexts/sizecontexts_wnids_test.csv', 'w') as f:
        for context in inds_contexts:
            csv.writer(f).writerow([ind2wnid[i] for i in context])
else:
    print('Suitable contexts not found')
