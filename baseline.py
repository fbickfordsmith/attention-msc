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
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white' # path to examples (should be in category folders)
batch_size = 89 # 48238=2*89*271
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False, # False => returns images in order
    class_mode=None) # None => returns just images (no labels)
true_top1 = generator.classes
num_examples = len(true_top1)
predicted_prob = model.predict_generator(
    generator,
    steps=num_examples//batch_size,
    verbose=True)
predicted_top1 = np.argmax(predicted_prob, axis=1)
accuracy = np.mean(predicted_top1==true_top1)
print(f'Top-1 accuracy: {(accuracy*100):.2f}%')
