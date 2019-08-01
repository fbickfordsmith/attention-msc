'''
Define an elementwise-multiplication attention layer.

References:
- github.com/keras-team/keras/blob/master/keras/constraints.py
- stackoverflow.com/questions/46821845/how-to-add-a-trainable-hadamard-product-layer-in-keras
- keras.io/layers/writing-your-own-keras-layers/
- tensorflow.org/beta/guide/keras/custom_layers_and_models
'''

from keras import backend as K
from keras.engine.topology import Layer
from keras.initializers import TruncatedNormal

class Constraint(object):
    def __call__(self, w):
        return w
    def get_config(self):
        return {}

class GreaterEqualEpsilon(Constraint):
    def __call__(self, w):
        w *= K.cast(K.greater_equal(w, K.epsilon()), K.floatx()) # W >= epsilon
        return w

class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1,) + input_shape[1:],
            initializer=TruncatedNormal(mean=1.0, stddev=0.1), # mean +/- 2 std
            trainable=True,
            constraint=GreaterEqualEpsilon())
        super(Attention, self).build(input_shape)

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
