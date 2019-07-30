import numpy as np
from sklearn.preprocessing import normalize
import time

path_activations = '/Users/fbickfordsmith/activations-copy/'
# path_activations = '/home/freddie/activations/'
path_save = '/home/freddie/attention/activations/'
num_samples = 200
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
                [np.tensordot(np.roll(Ai, shift, axis=0), Aj)/Ai.shape[0]
                for shift in range(Ai.shape[0])]))

np.save(f'{path_save}activations_dotproduct.npy', similarity, allow_pickle=False)

i = 30
d = []
Ai = normalize(A[i], axis=1)
start = time.time()
for j in range(1000):
    Aj = normalize(A[j], axis=1)
    d.append(np.mean(np.array(
        # [np.tensordot(np.roll(Ai, shift, axis=0), Aj)/Ai.shape[0]
        [np.sum(np.roll(Ai, shift, axis=0)*Aj, axis=1)
        for shift in range(num_samples)])))


print(time.time()-start)
print('hi')
# takes ~140 seconds per loop => 140,000/3600 = 39 hours but can halve that
