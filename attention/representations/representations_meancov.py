"""
For each ImageNet category, take the VGG16 representations of images in it
(computed using `representations_all.py`), and compute the mean and covariance
of these.

Time and memory requirements (using float32) for covariance:
- Full: ~10 seconds to fit; ~67 MB to store (4096x4096 matrix)
- Diagonal: ~0.2 seconds to fit; ~0.02 MB to store (4096x1 vector)
- Spherical: ~0 seconds to fit; ~0 MB to store (scalar)

We can get the spherical covariance by taking the mean of the diagonals of
of the diagonal covariance matrix.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import time
from ..utils.paths import path_activations, path_repo, path_representations

means, covariances = [], []
start = time.time()

for i in range(1000):
    if i % 100 == 0:
        print(f'i = {i}, time = {time.time()-start}')
    activations = np.load(
        path_activations/f'class{i:04}_activations.npy')
    gm = GaussianMixture(covariance_type='diag').fit(activations)
    means.append(gm.means_[0])
    covariances.append(gm.covariances_[0])

np.save(
    path_representations/'representations_mean.npy',
    np.array(means),
    allow_pickle=False)

np.save(
    path_representations/'representations_covariance.npy',
    np.array(covariances),
    allow_pickle=False)
