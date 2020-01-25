'''
For each pair of ImageNet classes, compute the similarity. Here, similarity is
measured by the dot product of VGG16 representations of sampled examples from
each class. (NumPy version)
'''

import numpy as np
from sklearn.preprocessing import normalize
import time

path_activations = '/Users/fbickfordsmith/activations-copy/'
path_save = '/Users/fbickfordsmith/Google Drive/Project/attention/activations/'
# path_activations = '/home/freddie/activations/'
# path_save = '/home/freddie/attention/activations/'
num_samples = 125
similarity = np.zeros((1000, 1000))

A = []
for i in range(1000):
    a = np.load(f'{path_activations}class{i:04}_activations.npy')
    A.append(a[np.random.randint(a.shape[0], size=num_samples)])
A = np.array(A) # (1000, num_samples, 4096)

start = time.time()
for i in range(1000):
    Ai = normalize(A[i])
    print(f'i = {i}, time = {int(time.time()-start)} seconds')
    for j in range(1000):
        if j >= i:
            Aj = normalize(A[j])
            similarity[i, j] = np.mean(np.array(
                [np.tensordot(np.roll(Ai, shift, axis=0), Aj)/num_samples
                for shift in range(num_samples)]))

np.save(f'{path_save}activations_dotproduct.npy', similarity, allow_pickle=False)
