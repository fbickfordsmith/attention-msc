import numpy as np
from sklearn.preprocessing import normalize
import time

path_activations = '/Users/fbickfordsmith/activations-copy/'
path_save = '/Users/fbickfordsmith/Google Drive/Project/attention/activations/'
num_samples = 50
dotproducts = np.zeros((1000, 1000))
start = time.time()

for i in range(1000):
    if i % 10 == 0: print(f'i = {i}, time = {int(time.time()-start)} seconds')
    Ai0 = normalize(np.load(f'{path_activations}class{i:04}_activations.npy'))
    Ai1 = Ai0[np.random.randint(Ai0.shape[0], size=num_samples)]
    for j in range(1000):
        if j >= i:
            Aj0 = normalize(np.load(f'{path_activations}class{j:04}_activations.npy'))
            Aj1 = Aj0[np.random.randint(Aj0.shape[0], size=num_samples)]
            dotproducts[i, j] = np.mean(np.array(
                [np.tensordot(np.roll(Ai1, shift, axis=0), Aj1)
                for shift in range(Ai1.shape[0])]))

np.save(f'{path_save}activations_dotproduct.npy', dotproducts, allow_pickle=False)
