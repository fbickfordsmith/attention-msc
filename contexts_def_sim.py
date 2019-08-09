from contexts_definition import *

num_seeds = 5
context_size = 50
inds_end = np.linspace(50, 999, 4, dtype=int)
interval_ends = np.arange(0.1, 0.65, 0.05) # interval_ends = np.arange(30, 85, 5)

intervals_covered = False
min_std_acc = np.inf
inds_keep = None
for _ in range(10000):
    inds_contexts, dists, accs = [], [], []
    # inds_seed = np.random.choice(inds_av_acc, size=num_seeds, replace=False)
    inds_seed = np.random.choice(1000, size=num_seeds, replace=False)
    for ind_seed in inds_seed:
        inds_sorted = np.argsort(Xdist[ind_seed])[1:] # 1 => don't include seed index
        for ind_end in inds_end:
            inds_sampled = np.random.choice(
                inds_sorted[:ind_end], size=context_size-1, replace=False)
            inds_sampled = np.insert(inds_sampled, 0, ind_seed)
            inds_contexts.append(inds_sampled)
            dists.append(average_distance(distances(X[inds_sampled])))
            accs.append(acc_score(inds_sampled))
    intervals_covered = check_coverage(np.array(dists), interval_ends)
    std_acc_sampled = np.std(accs)
    if intervals_covered and (std_acc_sampled < min_std_acc):
        inds_keep = inds_contexts
        min_std_acc = std_acc_sampled

if inds_keep is not None:
    print(min_std_acc)
    wnids_contexts = np.vectorize(ind2wnid.get)(inds_keep)
    pd.DataFrame(wnids_contexts).to_csv(
        f'{path}contexts/simcontexts_wnids.csv', header=False, index=False)
        # f'{path}contexts/simcontexts_wnids_test.csv', header=False, index=False)
else:
    print('Suitable contexts not found')
