'''
References:
- nealjean.com/ml/frechet-inception-distance/
- stats.stackexchange.com/questions/181620/what-is-the-meaning-of-super-script-2-subscript-2-within-the-context-of-norms
- core.ac.uk/download/pdf/82269844.pdf
'''

import numpy as np
from scipy.linalg import sqrtm
import time

path_activations = '/Users/fbickfordsmith/love16/activations/'
path_save = '/Users/fbickfordsmith/Google Drive/Project/attention/npy/'
frechet = np.empty((1000, 1000))

for i in range(1000):
    for j in range(1000):
        # shape(Z_i) = (num_examples_in_class_i, 4096)
        # Z_i[k] = VGG representation of example k from class i
        Z_i = np.load(f'{path_activations}class{i:04}_activations.npy')
        Z_j = np.load(f'{path_activations}class{j:04}_activations.npy')
        mean_i = np.mean(Z_i, axis=0)
        mean_j = np.mean(Z_j, axis=0)
        cov_i = np.diag(np.std(Z_i, axis=0))
        cov_j = np.diag(np.std(Z_j, axis=0))

        # compute the frechet distance
        frechet[i, j] = (
            np.sum((mean_i - mean_j)**2) +
            np.trace(cov_i + cov_j - 2*(sqrtm(np.dot(cov_i, cov_j)))))

np.save(f'{path_save}frechet.npy', frechet, allow_pickle=False)
