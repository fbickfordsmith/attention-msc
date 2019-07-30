import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
import time

path_activations = '/home/freddie/activations/'
path_save = '/home/freddie/attention/activations/'
num_samples = 700
similarity, A_np = [], []

for i in range(1000):
    a = np.load(f'{path_activations}class{i:04}_activations.npy')
    A_np.append(a[np.random.randint(a.shape[0], size=num_samples)])

A_np = np.array(A_np).transpose(1, 2, 0) # (1000, num_samples, 4096) -> (num_samples, 4096, 1000)
start = time.time()

for i in range(1000):
    print(f'i = {i}, time = {int(time.time()-start)} seconds')
    tf.reset_default_graph()
    X0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096))
    Y0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096, 1000))
    X = tf.math.l2_normalize(X0, axis=1)
    Y = tf.math.l2_normalize(Y0, axis=1)
    Z = tf.reduce_mean(tf.convert_to_tensor(
            [tf.reduce_sum(tf.multiply(tf.expand_dims(tf.roll(X, shift, axis=0), -1), Y), axis=1)
            for shift in range(num_samples)]), axis=(0, 1))
    z = tf.Session().run(Z, feed_dict={X0:A_np[:, :, i], Y0:A_np})
    similarity.append(z)

np.save(f'{path_save}activations_cosine.npy', np.array(similarity), allow_pickle=False)
