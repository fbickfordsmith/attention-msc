'''
Check that an attention model with unit attention weights (ie, there is an
attention layer between the final conv layer and the first FC layer, but all
weights are set to one) achieves the same performance as a VGG16 without an
attention layer.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
from attention_layer import GreaterEqualEpsilon

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
    layer.trainable=False

print('\nAttention model layers:')
for i in list(enumerate(attention_model.layers)):
    print(i)
print('\nTrainable weights:')
for i in attention_model.trainable_weights:
    print(i)
print('\n')

attention_model.compile(
    optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/'
batch_size = 256
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')

scores = model.evaluate_generator(
    generator,
    steps=int(np.ceil(generator.n/generator.batch_size)),
    use_multiprocessing=False,
    verbose=True)

print(f'{model.metrics_names} = {scores}')
