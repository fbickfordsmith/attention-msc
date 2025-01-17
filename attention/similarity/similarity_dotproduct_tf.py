"""
For each pair of ImageNet categories, compute the similarity. Here, similarity
is measured by the dot product of VGG16 representations of sampled examples from
each category. (TensorFlow version)

References:
- stackoverflow.com/questions/43357732/how-to-calculate-the-cosine-similarity-between-two-tensors/43358711
"""

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import time
import numpy as np
import tensorflow as tf
from ..utils.paths import path_activations, path_representations

num_samples = 540
num_splits = 3

similarity, A_np = [], []

for i in range(1000):
    a = np.load(path_activations/f'class{i:04}_activations.npy')
    A_np.append(a[np.random.randint(a.shape[0], size=num_samples)])

A_np = np.array(A_np).transpose(1, 2, 0) # A_np.shape = (num_samples, 4096, 1000)

start = time.time()

for i in range(1000):
    print(f'i = {i}, time = {int(time.time() - start)} seconds')
    z = np.zeros(1000)
    for j in range(num_splits):
        tf.reset_default_graph()
        X0 = tf.placeholder(
            dtype=tf.float32, shape=(num_samples//num_splits, 4096))
        Y0 = tf.placeholder(
            dtype=tf.float32, shape=(num_samples//num_splits, 4096, 1000))
        X = tf.math.l2_normalize(X0, axis=1)
        Y = tf.math.l2_normalize(Y0, axis=1)
        Z = tf.reduce_mean(
                tf.convert_to_tensor([
                    tf.reduce_sum(
                        tf.multiply(
                            tf.expand_dims(
                                tf.roll(X, shift, axis=0), -1), Y), axis=1)
                    for shift in range(num_samples)]),
                    axis=(0, 1))
        z += tf.Session().run(Z, feed_dict={
            # X0:A_np[:, :, i], Y0:A_np})
            X0:A_np[j*(num_samples//num_splits):(j+1)*(num_samples//num_splits), :, i],
            Y0:A_np[j*(num_samples//num_splits):(j+1)*(num_samples//num_splits)]})
    similarity.append(z)

np.save(
    path_representations/'representations_dotproduct.npy',
    np.array(similarity),
    allow_pickle=False)
