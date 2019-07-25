'''
References:
- https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
- https://stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
'''

import numpy as np
import time

path = '/Users/fbickfordsmith/Google Drive/Project/attention/npy/'
means = np.load(path+'activations_mean.npy')
covariances = np.load(path+'activations_cov.npy')
divergences = np.empty((1000, 1000))

def kl(m0, S0, m1, S1):
    # S0 and S1 are vectors
    k = m0.shape[0]
    iS1 = np.diag(1/S1)
    diff = m1 - m0
    tr_term = np.sum(S0/S1)
    logdetS0 = np.sum(np.log(S0))
    logdetS1 = np.sum(np.log(S1))
    det_term = logdetS1 - logdetS0
    quad_term = diff.T @ iS1 @ diff
    return 0.5 * (tr_term + det_term + quad_term - k)

start = time.time()
for i in range(1000):
    if i % 100 == 0:
        print(f'i = {i}, time={time.time()-start}')
    for j in range(1000):
        divergences[i, j] = kl(means[i], covariances[i], means[j], covariances[j])

np.save(f'{path}kl_divergence.npy', divergences, allow_pickle=False)
