'''
For each image in the ImageNet training set, get the VGG16 representation at the
final convolutional layer (ie, the activation of the layer just before the
fully-connected section).
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.layers import Input
from testing import predict_model

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
path_activations = '/home/freddie/activations-conv/'

vgg = VGG16(weights='imagenet')
input = Input(batch_shape=(None, 224, 224, 3))
output = vgg.layers[1](input)
for layer in vgg.layers[2:19]:
    output = layer(output)
model = Model(input, output)
activations, generator = predict_model(model, path_data)

# need to split this up to limit memory usage

for i in range(generator.num_classes):
    class_activations = activations[np.flatnonzero(generator.classes==i)].astype(np.float32)
    np.save(
        f'{path_activations}class{i:04}_activations_conv.npy',
        class_activations,
        allow_pickle=False)
