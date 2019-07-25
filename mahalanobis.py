'''
References:
- en.wikipedia.org/wiki/Mahalanobis_distance
- docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
'''

import numpy as np
from scipy.spatial.distance import cdist
import time

path = '/Users/fbickfordsmith/Google Drive/Project/attention/npy/'
means = np.load(path+'activations_mean.npy')
covariances = np.load(path+'activations_cov.npy')
mahalanobis = []
start = time.time()

for i in range(1000):
    if i % 100 == 0:
        print(f'i = {i}, time={time.time()-start}')
    mahalanobis.append(cdist(means, means[i][None, :], VI=np.diag(1/covariances[i]), metric='mahalanobis'))

# mahalanobis[r][c] = distance(mean_c, distribution_r)
# => np.array(mahalanobis).T[r, c] = distance(mean_r, distribution_c)
np.save(f'{path}mahalanobis.npy', np.array(mahalanobis).T, allow_pickle=False)
