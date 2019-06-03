import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Use a pretrained VGG16 to classify ImageNet examples.
Find the accuracy for each category.

Reference: https://medium.com/@vijayabhaskar96/tutorial-image-classification-
    with-keras-flow-from-directory-and-generators-95f75ebe5720
'''

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

model = VGG16(weights='imagenet')
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white' # path to examples (should be in category folders)
batch_size = 178 # 48238=2*89*271
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False, # False => returns images in order
    class_mode=None) # None => returns just images (no labels)
true_classes = generator.classes
num_examples = len(true_classes)
predicted_probs = model.predict_generator(
    generator,
    steps=num_examples//batch_size,
    verbose=True)
predicted_classes = np.argmax(predicted_probs, axis=1)
correct = (predicted_classes==true_classes)
accuracy = np.mean(correct)
print(f'Top-1 accuracy: {(accuracy*100):.2f}%')

ind_correct = np.flatnonzero(correct)
classes_correct = true_classes[ind_correct]
# Alternative/equivalent method
# classes_correct = (correct * (true_classes+1)) - 1 # -1 if incorrect; else class

name2ind = generator.class_indices
class_indices = np.arange(1000)
class_names = list(name2ind.keys())
counts_classes = np.array(
    [np.count_nonzero(true_classes==i) for i in range(1000)])
counts_correct = np.array(
    [np.count_nonzero(classes_correct==i) for i in range(1000)])
# accuracy_classes = counts_correct / counts_classes
results = np.stack(
    (class_indices, counts_classes, counts_correct), axis=1)
np.savetxt('classwise_accuracy.csv', results)
np.savetxt('class_names.csv', np.array(class_names), fmt='%s')

# Code for converting indices to class IDs
# name2ind = generator.class_indices
# ind2name = {ind:name for name, ind in name2ind.items()}
# predicted_names = [ind2name[ind] for ind in predicted_labels]
