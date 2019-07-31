'''
Assess the accuracy of a pretrained VGG16 on the ImageNet validation set, using
the preprocessing routine defined in img_processing.py.

Command-line arguments:
1. data_partition in {train, val, val_white}
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from img_processing import robinson_processing

_, data_partition = sys.argv
path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
datagen = ImageDataGenerator(preprocessing_function=robinson_processing)

generator = datagen.flow_from_directory(
    directory=path_data,
    target_size=(224, 224),
    batch_size=256,
    shuffle=False,
    class_mode='categorical')

scores = model.evaluate_generator(
    generator,
    steps=int(np.ceil(generator.n/generator.batch_size)),
    use_multiprocessing=False,
    verbose=True)

print(f'{model.metrics_names} = {scores}')
