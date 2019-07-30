import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
import tensorflow as tf
import time

# path_activations = '/Users/fbickfordsmith/activations-copy/'
path_activations = '/home/freddie/activations/'
path_save = '/home/freddie/attention/activations/'
num_samples = 700
similarities = np.zeros((1000, 1000))

A_np = []
for i in range(1000):
    a = np.load(f'{path_activations}class{i:04}_activations.npy')
    A_np.append(a[np.random.randint(a.shape[0], size=num_samples)])

A_np = np.array(A_np).transpose(1, 2, 0) # (1000, num_samples, 4096) -> (num_samples, 4096, 1000)

tf.reset_default_graph()
X0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096))
Y0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096, 1000))
X = tf.math.l2_normalize(X0, axis=1)
Y = tf.math.l2_normalize(Y0, axis=1)
Z = tf.reduce_mean(tf.convert_to_tensor(
    [tf.reduce_sum(tf.multiply(tf.expand_dims(tf.roll(X, shift, axis=0), -1), Y), axis=1)
    for shift in range(num_samples)]), axis=(0, 1))

start = time.time()
with tf.Session() as sess:
    z = sess.run(Z, feed_dict={X0:A_np[:, :, 30], Y0:A_np})


print(time.time()-start)
print('hi')

# tf.reset_default_graph()
# X0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096))
# Y0 = tf.placeholder(dtype=tf.float32, shape=(num_samples, 4096, 1000))
# X = tf.math.l2_normalize(X0, axis=1)
# Y = tf.math.l2_normalize(Y0, axis=1)
# a = tf.multiply(tf.expand_dims(tf.roll(X, shift, axis=0), -1), Y)
# b = tf.reduce_sum(a, axis=1)
# c = tf.reduce_mean(b, axis=0)
# shift = 3
# i, j = 30, 40
# with tf.Session() as sess:
#     e0 = sess.run(c, feed_dict={X0:A_np[:, :, i], Y0:A_np})

# d = []
# for shift in range(num_samples):
#     a = tf.multiply(tf.expand_dims(tf.roll(X, shift, axis=0), -1), Y)
#     b = tf.reduce_sum(a, axis=1)
#     c = tf.reduce_mean(b, axis=0)
#     d.append(c)
#
# e = tf.reduce_mean(tf.convert_to_tensor(d))
# with tf.Session() as sess:
    # e0 = sess.run(e, feed_dict={X0:A_np[:, :, i], Y0:A_np})
