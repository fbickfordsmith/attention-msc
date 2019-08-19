'''
Method:
...
For each seed
- For k in {50, 366, 682, 999}, sample 49 indices (\seed) from the k nearest neighbours

Previous versions used
- random.choice(inds_av_acc) instead of random.choice(1000)
- Euclidean distance with interval_ends = np.arange(30, 85, 5)
'''

from contexts_definition import *

type_context = 'sim'
version_wnids = 'v' + input('Version number (WNIDs): ')

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
    print('Distance:', [round(average_distance(distances(Z[inds])), 2) for inds in inds_best])
    wnids_best = np.vectorize(ind2wnid.get)(inds_best)
    pd.DataFrame(wnids_best).to_csv(
        f'{path}contexts/{type_context}_{version_wnids}_wnids.csv', header=False, index=False)
else:
    print('Suitable contexts not found')
