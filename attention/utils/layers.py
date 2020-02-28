"""
Define an elementwise-multiplication attention layer.

References:
- github.com/keras-team/keras/blob/master/keras/constraints.py
- stackoverflow.com/questions/46821845/how-to-add-a-trainable-hadamard-product-layer-in-keras
- keras.io/layers/writing-your-own-keras-layers/
- tensorflow.org/beta/guide/keras/custom_layers_and_models
"""

import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1,)+input_shape[1:],
            initializer='ones',
            trainable=True,
            constraint=tf.keras.constraints.NonNeg())

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
