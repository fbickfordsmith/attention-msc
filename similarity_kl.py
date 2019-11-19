'''
For each pair of ImageNet classes, compute the similarity. Here, similarity is
measured by the KL divergence of VGG16 representations for each class.

References:
- en.wikipedia.org/wiki/Kullback–Leibler_divergence
- stackoverflow.com/questions/44549369/kullback-leibler-divergence-from-gaussian-pm-pv-to-gaussian-qm-qv
'''

import numpy as np
import time

path = '/Users/fbickfordsmith/Google Drive/Project/attention/representations/'
means = np.load(path+'representations_mean.npy')
covariances = np.load(path+'representations_covariance.npy')
divergences = np.empty((1000, 1000))

def kl(m0, S0, m1, S1):
    # S0 and S1 are vectors of the diagonals of covariance matrices
    k = m0.shape[0]
    iS1 = 1/S1 # to invert a diagonal matrix, take reciprocals of the diagonals
    diff = m1 - m0
    tr_term = np.sum(S0*iS1)
    logdetS0 = np.sum(np.log(S0))
    logdetS1 = np.sum(np.log(S1))
    det_term = logdetS1 - logdetS0
    quad_term = diff.T @ np.diag(iS1) @ diff
    return 0.5 * (tr_term + det_term + quad_term - k)

start = time.time()
for i in range(1000):
    if i % 10 == 0:
        print(f'i = {i}, time={time.time()-start}')
    for j in range(1000):
        divergences[i, j] = kl(means[i], covariances[i], means[j], covariances[j])

np.save(f'{path}kl_divergence.npy', divergences, allow_pickle=False)
