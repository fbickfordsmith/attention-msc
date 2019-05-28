import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white' # path to examples (should be in category folders)
batch_size = 1000 # change to match memory capacity
model = VGG16(weights='imagenet')
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode=None) # no labels are returned; generator just produces batches
predictions = model.predict_generator(generator, steps=50)
