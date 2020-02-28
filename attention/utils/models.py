"""
Take a pretrained VGG16. Add an elementwise-multiplication attention layer
between the final convolutional layer and the first fully-connected layer. Fix
all weights except for the attention weights.
"""

import numpy as np
import tensorflow as tf
from ..utils.layers import Attention

def build_model(attention_layer=Attention(), train=True, attention_position=19):
    vgg = tf.keras.applications.vgg16.VGG16()
    input = tf.keras.layers.Input(shape=(224, 224, 3))
    output = vgg.layers[1](input)
    for layer in vgg.layers[2:attention_position]:
        output = layer(output)
    output = attention_layer(output)
    for layer in vgg.layers[attention_position:]:
        output = layer(output)
    model = tf.keras.models.Model(input, output)
    for layer in model.layers:
        if ('attention' in layer.name) and (train == True):
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=3e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])
    return model
