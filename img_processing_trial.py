import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from image_processing import robinson_processing

model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/'
partition = sys.argv[1] # one of [train, val, val_white]
batch_size = 256
datagen = ImageDataGenerator(preprocessing_function=robinson_processing)

generator = datagen.flow_from_directory(
    directory=path+partition,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')

scores = model.evaluate_generator(
    generator,
    steps=int(np.ceil(generator.n/generator.batch_size)),
    use_multiprocessing=False,
    verbose=True)

print(f'{model.metrics_names} = {scores}')
