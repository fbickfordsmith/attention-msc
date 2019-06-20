'''
Assess the accuracy of a pretrained VGG16 on the ImageNet validation set.

References:
- medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

model = VGG16(weights='imagenet')
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/' # path to examples (should be in category folders)
batch_size = 256 # 48238=2*89*271
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator_pred = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False, # False => returns images in order
    class_mode=None) # None => returns just images (no labels)
predicted_prob = model.predict_generator(
    generator_pred,
    steps=int(np.ceil(generator_pred.n/generator_pred.batch_size)),
    use_multiprocessing=False,
    verbose=True)
predicted_top1 = np.argmax(predicted_prob, axis=1)
true_top1 = generator_pred.classes
accuracy = np.mean(predicted_top1==true_top1)
print(f'Using predict_generator, accuracy = {accuracy}') # {(accuracy*100):.2f}%')

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
