'''
Test the generator defined in generator.py.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from keras.applications.vgg16 import VGG16
from image_processing import robinson_processing
from keras.applications.vgg16 import VGG16, preprocess_input
from generator import build_generators

model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])

test_generator = build_generators(
    path_data='/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/',
    path_synsets='~/vgg16/txt/synsets.txt',
    batch_size=256,
    preprocess_fn=robinson_processing)

score = model.evaluate_generator(
    test_generator,
    steps=test_generator.__len__(),
    use_multiprocessing=True,
    workers=7,
    verbose=True)

print(model.metrics_names)
print(score)
