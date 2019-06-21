'''
For each ImageNet class, assess the accuracy of a pretrained VGG16 on the
ImageNet validation set.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

model = VGG16(weights='imagenet')
path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white' # path to examples (should be in category folders)
batch_size = 256
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False, # False => returns images in order
    class_mode=None) # None => returns just images (no labels)

probability = model.predict_generator(
    generator,
    steps=int(np.ceil(generator.n/generator.batch_size)),
    use_multiprocessing=False,
    verbose=True)

predicted_class = np.argmax(probability, axis=1)
true_class = generator.classes
correct_bool = (predicted_top1==true_top1)
correct_class = true_class[np.flatnonzero(correct_bool)] # vector where each entry is the class of an example that has been correctly classified

df = pd.DataFrame()
df['wnid'] = generator.class_indices.keys() # generator.class_indices is a wnid:index dictionary
df['num_examples'] = [np.count_nonzero(true_class==i) for i in range(1000)]
df['num_correct'] = [np.count_nonzero(correct_class==i) for i in range(1000)]
df['accuracy'] = df['num_correct'] / df['num_examples']
df.to_csv('csv/baseline_classwise.csv')

# class_indices = np.arange(1000)
# class_names = list(name2ind.keys())
# counts_class = np.array(
#     [np.count_nonzero(true_class==i) for i in range(1000)])
# counts_correct = np.array(
#     [np.count_nonzero(class_correct==i) for i in range(1000)])
# accuracy_class = counts_correct / counts_class
# results = np.stack(
#     (class_indices, counts_class, counts_correct), axis=1)
# np.savetxt('classwise_accuracy.csv', results)
# np.savetxt('class_names.csv', np.array(class_names), fmt='%s')

# Code for converting indices to class IDs
# name2ind = generator.class_indices
# ind2name = {ind:name for name, ind in name2ind.items()}
# predicted_names = [ind2name[ind] for ind in predicted_labels]
