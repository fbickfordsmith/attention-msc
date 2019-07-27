'''
For each ImageNet class, find the mean and covariance of the VGG16
representations of images in it.
'''

# sshfs freddie@love16.pals.ucl.ac.uk:/home/freddie /Users/fbickfordsmith/love16
# rsync --progress --recursive '/Users/fbickfordsmith/love16/activations/' '/Users/fbickfordsmith/activations-copy/'

import numpy as np
from sklearn.mixture import GaussianMixture
import time

path_activations = '/Users/fbickfordsmith/activations-copy/'
path_save = '/Users/fbickfordsmith/Google Drive/Project/attention/npy/'
means, covariances = [], []
start = time.time()

for i in range(1000):
    if i % 100 == 0:
        print(f'i = {i}, time={time.time()-start}')
    activations = np.load(
        f'{path_activations}class{i:04}_activations.npy')

    # full cov takes ~10 seconds per fit and ~134 MB to store
    # diag cov takes ~0.2 seconds per fit and ~0.03 MB to store
    gm = GaussianMixture(covariance_type='diag').fit(activations)
    means.append(gm.means_[0])
    covariances.append(gm.covariances_[0])

np.save(f'{path_save}mean_activations', np.array(means), allow_pickle=False)
np.save(f'{path_save}cov_activations', np.array(covariances), allow_pickle=False)

# if we want the spherical cov, take the mean of the diag cov,
