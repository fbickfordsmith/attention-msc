'''
Old version:
def frechet(m0, S0, m1, S1):
    # S0 and S1 are vectors of the diagonals of covariance matrices
    S0, S1 = np.diag(S0), np.diag(S1)
    return (
        np.sum((m0 - m1)**2) +
        np.trace(S0 + S1 - 2*(sqrtm(np.dot(S0, S1)))))

Changes:
    tr(A + B) = tr(A) + tr(B)
    np.dot(S0, S1) = np.diag(S0*S1)
    sqrtm(np.diag(S0*S1)) = np.diag(np.sqrt(S0*S1)) if S0 and S1 are nonnegative

References:
- nealjean.com/ml/frechet-inception-distance/
- stats.stackexchange.com/questions/181620/what-is-the-meaning-of-super-script-2-subscript-2-within-the-context-of-norms
- core.ac.uk/download/pdf/82269844.pdf
- math.stackexchange.com/questions/2965025/square-root-of-diagonal-matrix
- djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/#eqWG
- stats.stackexchange.com/questions/83741/earth-movers-distance-emd-between-two-gaussians
'''

import numpy as np
# from scipy.linalg import sqrtm
import time

path = '/Users/fbickfordsmith/Google Drive/Project/attention/npy/'
means = np.load(path+'activations_mean.npy')
covariances = np.load(path+'activations_cov.npy')
distances = np.empty((1000, 1000))

def frechet(m0, S0, m1, S1):
    # S0 and S1 are vectors of the diagonals of covariance matrices
    return np.sum((m0-m1)**2) + np.sum(S0+S1) - np.sum(2*np.sqrt(S0*S1))

start = time.time()
for i in range(1000):
    if i % 100 == 0:
        print(f'i = {i}, time={time.time()-start}')
    for j in range(1000):
        distances[i, j] = frechet(
            means[i], covariances[i], means[j], covariances[j])

np.save(f'{path}frechet.npy', distances, allow_pickle=False)
