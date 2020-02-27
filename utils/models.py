"""
Take a pretrained VGG16. Add an elementwise-multiplication attention layer
between the final convolutional layer and the first fully-connected layer. Fix
all weights except for the attention weights.
"""

import sys
sys.path.append('..')

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from utils.layers import Attention

def build_model(attention_layer=Attention(), train=True, attention_position=19):
    vgg = VGG16()
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
        optimizer=Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    print(
        '\nLayers:', *enumerate(model.layers),
        '\nTrainable weights:', *model.trainable_weights, '', sep='\n')
    return model
