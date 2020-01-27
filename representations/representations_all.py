"""
For each image in the ImageNet training set, get the VGG16 representation at the
penultimate layer (ie, the activation of the layer just before the softmax).
"""

import os, sys
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
sys.path.append('..')

import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from utils.testing import predict_model
from utils.paths import path_imagenet, path_activations

path_data = path_imagenet/'train/'

# Copy all layers except for the final one
vgg = VGG16()
input = Input(batch_shape=(None, 224, 224, 3))
output = vgg.layers[1](input)
for layer in vgg.layers[2:-1]:
    output = layer(output)
model = Model(input, output)
activations, generator = predict_model(model, path_data)

# Need to split this up to limit memory usage
for i in range(generator.num_classes):
    class_activations = activations[np.flatnonzero(generator.classes==i)]
    np.save(
        path_activations/f'class{i:04}_activations.npy',
        class_activations,
        allow_pickle=False)
