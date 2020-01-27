"""
Define a set of 20 'similarity contexts'. These are subsets of ImageNet classes
that we choose to have varying visual similarity (average pairwise cosine
similarity of VGG16 representations) but equal size and approx equal difficulty.

Method:
1. Sample 5 seeds.
2. For each seed,
    a. For k in {50, 366, 682, 999},
        i.  Uniformly sample 49 indices from the seed's k nearest neighbours.
        ii. Compute the distance (= 1 - similarity) and accuracy of the sampled
            context.
3. Check that the sampled contexts give good coverage of similarity values
    between 0.1 and 0.6.
4. Keep the sampled contexts if their accuracy score (normalised distance from
    average VGG16 accuracy) is better than any previous score.
"""

version_wnids = input('Version number (WNIDs): ')

import sys
sys.path.append('..')

from utils.paths import path_repo
from utils.contexts_definition import *
from utils.metadata import ind2wnid

path_save = path_repo/f'data/contexts/sim_v{version_wnids}_wnids.csv'

num_seeds = 5
context_size = 50
inds_end = np.linspace(50, 999, 4, dtype=int)
interval_ends = np.arange(0.1, 0.65, 0.05)

intervals_covered = False
acc_bestscore = np.inf
inds_best = None

for i in range(10000):
    if i % 1000 == 0:
        print(f'i = {i:05}')
    inds_sampled, dist, acc = [], [], []
    inds_seed = np.random.choice(1000, size=num_seeds, replace=False)
    for ind_seed in inds_seed:
        inds_sorted = np.argsort(Zdist[ind_seed])[1:] # 1 => don't include seed index
        for ind_end in inds_end:
            inds_context = np.random.choice(
                inds_sorted[:ind_end], size=context_size-1, replace=False) # 'probabilistic/sampled nearest neighbour'
            inds_context = np.insert(inds_context, 0, ind_seed)
            inds_sampled.append(inds_context)
            dist.append(average_distance(distances(Z[inds_context])))
            acc.append(score_acc(inds_context))
    intervals_covered = check_coverage(np.array(dist), interval_ends)
    acc_score = np.max(np.abs(acc)) # similar results with acc_score = np.std(acc)
    if intervals_covered and (acc_score < acc_bestscore):
        inds_best = inds_sampled
        acc_bestscore = acc_score

if inds_best is not None:
    print('Accuracy:', [round(base_accuracy(inds), 2) for inds in inds_best])
    print('Distance:',
        [round(average_distance(distances(Z[inds])), 2) for inds in inds_best])
    wnids_best = np.vectorize(ind2wnid.get)(inds_best)
    pd.DataFrame(wnids_best).to_csv(path_save, header=False, index=False)
else:
    print('Suitable contexts not found')
