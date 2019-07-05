'''
For each ImageNet class, assess the accuracy of a pretrained VGG16 on the
ImageNet validation set.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from testing import predict_model

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/'
model = VGG16(weights='imagenet')
# _, classes_pred, classes_true, wnid2ind = predict_model(model, path_data)
probabilities, generator = predict_model(model, path_data)
classes_pred = np.argmax(probabilities, axis=1)
classes_true = generator.classes
wnid2ind = generator.class_indices
correct_bool = (classes_pred==classes_true)
correct_class = classes_true[np.flatnonzero(correct_bool)] # vector where each entry is the class of an example that has been correctly classified

df = pd.DataFrame()
df['wnid'] = wnid2ind.keys()
df['num_examples'] = [np.count_nonzero(classes_true==i) for i in range(1000)]
df['num_correct'] = [np.count_nonzero(correct_class==i) for i in range(1000)]
df['accuracy'] = df['num_correct'] / df['num_examples']
df.to_csv('results/baseline_classwise_acc.csv')
