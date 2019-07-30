import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf
import time

# path_activations = '/Users/fbickfordsmith/activations-copy/'
path_activations = '/home/freddie/activations/'
path_save = '/home/freddie/attention/activations/'
num_samples = 200
dotproducts = np.zeros((1000, 1000))

A_np = []
for i in range(1000):
    a = np.load(f'{path_activations}class{i:04}_activations.npy')
    A_np.append(a[np.random.randint(a.shape[0], size=num_samples)])

# np.array(A_np) is (1000, num_samples, 4096)
A_np = normalize(np.array(A_np).transpose(1, 2, 0), axis=2)

i, j = 30, 40

tf.reset_default_graph()
Ai0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096))
Aj0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096))
# Ai1 = tf.math.l2_normalize(Ai0, axis=1)
# Aj1 = tf.math.l2_normalize(Aj0, axis=1)
# d0 = tf.reduce_mean(tf.reduce_sum(tf.multiply(Ai1, Aj1), axis=1))
# d0 = tf.divide(tf.tensordot(Ai1, Aj1, axes=2), num_samples)
d0 = tf.metrics.mean_cosine_distance(Ai1, Aj1, dim=1)
d1 = tf.reduce_mean(tf.convert_to_tensor([
    # tf.metrics.mean_cosine_distance(tf.roll(Ai1, shift, axis=0), Aj1, dim=1)
    tf.divide(tf.tensordot(tf.roll(Ai1, shift, axis=0), Aj1, axes=2), num_samples)
    for shift in range(num_samples)]))

start = time.time()
with tf.Session() as sess:
    d = sess.run(d1, feed_dict={Ai0:A_np[i], Aj0:A_np[j]})

print(time.time()-start)
print('done')

print(d)
print(np.tensordot(normalize(A_np[i]), normalize(A_np[j])) / num_samples)

Ai = normalize(A_np[i])
Aj = normalize(A_np[j])
np.mean(np.array(
    [np.tensordot(np.roll(Ai, shift, axis=0), Aj)/Ai.shape[0]
    for shift in range(Ai.shape[0])]))
