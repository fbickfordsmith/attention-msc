import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Take a pretrained VGG16.
Add an attention layer between the final conv layer and the first FC layer.
Fix all parameters except for the attention weights.
Fix the attention weights to one.
Check performance.

References:
github.com/keras-team/keras/blob/master/keras/constraints.py
stackoverflow.com/questions/46821845/how-to-add-a-trainable-hadamard-product-layer-in-keras
stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
stackoverflow.com/questions/43906048/keras-early-stopping
'''

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.engine.topology import Layer
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.metrics import sparse_top_k_categorical_accuracy

################################################################################

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
            initializer='ones',
            trainable=True,
            constraint=GreaterEqualEpsilon())
        super(Attention, self).build(input_shape)

    def call(self, x):
        return x * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape

################################################################################

pretrained_model = VGG16(weights='imagenet')
model_in = Input(batch_shape=(None, 224, 224, 3))
model_out = pretrained_model.layers[1](model_in)
for layer in pretrained_model.layers[2:19]:
    model_out = layer(model_out)
model_out = Attention()(model_out)
for layer in pretrained_model.layers[19:]:
    model_out = layer(model_out)
attention_model = Model(model_in, model_out)
for i, layer in enumerate(attention_model.layers):
    # if i != 19:
    #     layer.trainable=False
    layer.trainable=False

print('\nAttention model layers:')
for i in list(enumerate(attention_model.layers)):
    print(i)
print('\nTrainable weights:')
for i in attention_model.trainable_weights:
    print(i)
print('\n')

attention_model.compile(
    # optimizer=optimizers.Adam(lr=3e-4), # Karpathy default
    optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), # relatively low lr (could also try 1e-4)
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']) # top-1 and top5 acc

################################################################################

path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white' # path to examples (should be in category folders)
batch_size = 178 # 48238=2*89*271
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False, # False => returns images in order
    class_mode=None) # None => returns just images (no labels)
true_top1 = generator.classes
num_examples = len(true_top1)
predicted_prob = attention_model.predict_generator(
    generator,
    steps=num_examples//batch_size,
    verbose=True)
predicted_top1 = np.argmax(predicted_prob, axis=1)
accuracy = np.mean(predicted_top1==true_top1)
print(f'Top-1 accuracy: {(accuracy*100):.2f}%')
