'''
For each image in the ImageNet training set, get a VGG16 representation (ie, the
activation of the layer before the softmax).
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Input
from testing import predict_model

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
path_activations = '/home/freddie/activations/'
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# copy all layers except for the final one
vgg = VGG16(weights='imagenet')
input = Input(batch_shape=(None, 224, 224, 3))
output = vgg.layers[1](input)
for layer in vgg.layers[2:-1]:
    output = layer(output)
model = Model(input, output)
activations, generator = predict_model(model, path_data)

for i in range(generator.num_classes):
    class_activations = activations[np.flatnonzero(generator.classes==i)]
    np.save(f
        f'{path_activations}class{i:04}_activations.npy'
        class_activations,
        allow_pickle=False)
