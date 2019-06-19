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

# copy all layers except for the final one
pretrained_model = VGG16(weights='imagenet')
model_in = Input(batch_shape=(None, 224, 224, 3))
model_out = pretrained_model.layers[1](model_in)
for layer in pretrained_model.layers[2:22]:
    model_out = layer(model_out)
modified_model = Model(model_in, model_out)

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
path_activations = '/home/freddie/activations/'
batch_size = 256
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(
    directory=path_data,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False,
    class_mode=None)

activations = modified_model.predict_generator(
    generator,
    steps=int(np.ceil(generator.n/generator.batch_size)),
    use_multiprocessing=True,
    workers=7,
    verbose=True)

for i in range(generator.num_classes):
    class_activations = activations[np.flatnonzero(generator.classes==i)]
    np.save(
        path_activations+f'class{i:04}_activations'
        class_activations,
        allow_pickle=False)
