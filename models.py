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

def build_model(attention_layer, train=True, attention_position=19):
    vgg = VGG16(weights='imagenet')
    input = Input(batch_shape=(None, 224, 224, 3))
    output = vgg.layers[1](input)
    for layer in vgg.layers[2:attention_position]:
        output = layer(output)
    output = attention_layer(output)
    for layer in vgg.layers[attention_position:]:
        output = layer(output)
    model = Model(input, output)
    for layer in model.layers:
        if ('attention' in layer.name) and train:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']) # top-1 and top5 acc
    print(
        '\nLayers:', *enumerate(model.layers),
        '\nTrainable weights:', *model.trainable_weights, '', sep='\n')
    return model
