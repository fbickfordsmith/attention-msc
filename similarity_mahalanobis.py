'''
For each pair of ImageNet classes, compute the similarity. Here, similarity is
measured by the Mahalanobis distance of VGG16 representations for each class.

References:
- en.wikipedia.org/wiki/Mahalanobis_distance
- docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
'''

import numpy as np
from scipy.spatial.distance import cdist
import time

path = '/Users/fbickfordsmith/Google Drive/Project/attention/representations/'
means = np.load(path+'representations_mean.npy')
covariances = np.load(path+'representations_covariance.npy')
mahalanobis = []
start = time.time()

for i in range(1000):
    if i % 10 == 0:
        print(f'i = {i}, time={time.time()-start}')
    mahalanobis.append(cdist(means, means[i][None, :], VI=np.diag(1/covariances[i]), metric='mahalanobis'))

# np.array(mahalanobis).shape = (1000, 1000, 1) but we want (1000, 1000)
# mahalanobis[r, c] = distance(mean_c, distribution_r) but we want distance(mean_r, distribution_c)
mahalanobis_arr = np.array(mahalanobis)[:, :, 0].T # (1000, 1000, 1) -> (1000, 1000)
np.save(f'{path}mahalanobis.npy', mahalanobis_arr, allow_pickle=False)
