'''
Define a model with elementwise-multiplication attention layer as the only
trainable layer.
'''

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
from keras import optimizers

def build_model(attention_layer, train=True, optimizer=optimizers.Adam(lr=3e-4)):
    vgg = VGG16(weights='imagenet')
    input = Input(batch_shape=(None, 224, 224, 3))
    output = vgg.layers[1](input)
    for layer in vgg.layers[2:19]:
        output = layer(output)
    output = attention_layer(output)
    # output = Attention()(output)
    for layer in vgg.layers[19:]:
        output = layer(output)
    model = Model(input, output)

    for layer in model.layers:
        if ('attention' in layer.name) and train:
            layer.trainable = True
        else:
            layer.trainable = False
    # for i, layer in enumerate(model.layers):
    #     if i != 19:
    #         layer.trainable=False

    print('\nAttention model layers:')
    for layer in list(enumerate(model.layers)):
        print(layer)

    print('\nTrainable weights:')
    for weight in model.trainable_weights:
        print(weight)
    print()

    model.compile(
        optimizer=optimizer,
        # optimizer=optimizers.Adam(lr=3e-4), # Karpathy default
        # optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), # could also try 1e-4
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']) # top-1 and top5 acc

    return model
