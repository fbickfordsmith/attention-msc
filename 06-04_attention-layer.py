import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Take a pretrained VGG16.
Add an attention layer between the final conv layer and the first FC layer.
Fix all parameters except for the attention weights.
Train on ImageNet.

References:
github.com/keras-team/keras/blob/master/keras/constraints.py
stackoverflow.com/questions/46821845/how-to-add-a-trainable-hadamard-product-layer-in-keras
stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
'''

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.engine.topology import Layer
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

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
            initializer='uniform',
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
    if i != 19:
        layer.trainable=False

print('\nAttention model layers:')
for i in list(enumerate(attention_model.layers)):
    print(i)
print('\nTrainable weights:')
for i in attention_model.trainable_weights:
    print(i)

attention_model.compile(
    # optimizer=optimizers.Adam(lr=3e-4), # Karpathy default
    optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), # relatively low lr (maybe try 1e-4)
    loss='categorical_crossentropy')

path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/'
batch_size = 256 # VGG paper
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1)
train_generator = datagen.flow_from_directory(
    directory=path+'train/',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True, # False => returns images in order
    class_mode='categorical', # None => returns just images (no labels)
    subset='training')
validation_generator = datagen.flow_from_directory(
    directory=path+'train/',
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True, # False => returns images in order
    class_mode='categorical', # None => returns just images (no labels)
    subset='validation')
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=1, # number of epochs without improvement after which we stop
    verbose=True,
    restore_best_weights=True) # False => weights from last step are used
attention_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.n//batch_size,
    epochs=10,
    verbose=1,
    callbacks=[early_stopping],
    validation_data=validation_generator,
    validation_steps=validation_generator.n//batch_size)
