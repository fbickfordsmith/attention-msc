import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from layers import Attention

def build_vgg2(attention_layer=Attention(), train=True):
    vgg = VGG16(weights='imagenet')
    input = Input(batch_shape=(None, 7, 7, 512))
    output = attention_layer(input)
    for layer in vgg.layers[19:]:
        output = layer(output)
    model = Model(input, output)
    for layer in model.layers:
        if ('attention' in layer.name) and train:
            layer.trainable = True
        else:
            layer.trainable = False
    model.compile(
        optimizer=optimizers.Adam(lr=3e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy']) # top1 and top5 acc
    print(
        '\nLayers:', *enumerate(model.layers),
        '\nTrainable weights:', *model.trainable_weights, '', sep='\n')
    return model
