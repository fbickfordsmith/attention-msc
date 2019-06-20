import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from image_processing import robinson_processing

model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/' # path to examples (should be in category folders)
batch_size = 256
datagen = ImageDataGenerator(preprocessing_function=robinson_processing)
generator_eval = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical')
scores = model.evaluate_generator(
    generator_eval,
    steps=int(np.ceil(generator_eval.n/generator_eval.batch_size)),
    use_multiprocessing=False,
    verbose=True)
print(f'Using evaluate_generator, {model.metrics_names} = {scores}')
