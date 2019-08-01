'''
Take a pretrained VGG16, and add an elementwise-multiplication attention layer
between the final convolutional layer and the first fully-connected layer. Fix
all weights except for the attention weights.
'''

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
from keras import optimizers

compile_params = dict(
        optimizer=optimizers.Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']) # top-1 and top5 acc

def build_model(attention_layer, train=True):
    vgg = VGG16(weights='imagenet')
    input = Input(batch_shape=(None, 224, 224, 3))
    output = vgg.layers[1](input)
    for layer in vgg.layers[2:19]:
        output = layer(output)
    output = attention_layer(output)
    for layer in vgg.layers[19:]:
        output = layer(output)
    model = Model(input, output)
    for layer in model.layers:
        if ('attention' in layer.name) and train:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(**compile_params)
    print(
        '\nLayers:', *enumerate(model.layers),
        '\nTrainable weights:', *model.trainable_weights, '', sep='\n')
    return model
