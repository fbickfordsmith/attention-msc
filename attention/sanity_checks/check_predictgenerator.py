"""
Assess the accuracy of a pretrained VGG16 on the ImageNet validation set,
without using `predict_generator`. Written for a sanity check. Agreement with
the result produced by `baseline_average.py` implies `predict_generator` works
as expected.
"""

gpu = input('GPU: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input, decode_predictions)
from tensorflow.keras.preprocessing.image import (
    load_img, img_to_array, ImageDataGenerator)
from ..utils.paths import path_imagenet

path_data = path_imagenet/'val_white/'
model = VGG16()
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(
    directory=path_data,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False,
    class_mode=None)

true_classes = generator.classes
path_imgs = generator.filenames
name2ind = generator.class_indices
ind2name = {ind:name for name, ind in name2ind.items()}
true_names = np.array([ind2name[ind] for ind in true_classes])
predicted_classes = []

for i, path_img in enumerate(path_imgs):
    if i % 2000 == 0:
        print(f'{i:05} images processed')
    img = load_img(path_data+path_img, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predicted_classes.append(np.argmax(model.predict(img)))

predicted_names = np.array([ind2name[ind] for ind in predicted_classes])
correct = (predicted_names == true_names)
accuracy = np.mean(correct)
print(f'Top-1 accuracy: {(accuracy*100):.2f}%')
