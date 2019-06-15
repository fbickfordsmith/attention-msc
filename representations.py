import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Get a VGG16 representation (activation of layer before the softmax) for all
examples in the ImageNet training set.
'''

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

path_to_activations = '/home/freddie/activations/'
mean_activations = []
for i in range(1000):
    all_activations = np.load(
        path_to_activations+f'class{i:04}_activations.npy')
    mean_activations.append(np.mean(all_activations, axis=0))
mean_activations = np.array(mean_activations)
similarity = cosine_similarity(mean_activations)
np.save(path_to_activations+'mean_activations', mean_activations, allow_pickle=False)
np.save(path_to_activations+'cosine_similarity', similarity, allow_pickle=False)

# from math import ceil
# import numpy as np
# from keras.applications.vgg16 import VGG16, preprocess_input
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Model
# from keras.layers import Input
# from sklearn.metrics.pairwise import cosine_similarity
#
# pretrained_model = VGG16(weights='imagenet')
# model_in = Input(batch_shape=(None, 224, 224, 3))
# model_out = pretrained_model.layers[1](model_in)
# for layer in pretrained_model.layers[2:22]:
#     model_out = layer(model_out)
# modified_model = Model(model_in, model_out)
#
# path_to_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
# path_to_activations = '/home/freddie/activations/'
# batch_size = 256
# datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# generator = datagen.flow_from_directory(
#     directory=path_to_data,
#     target_size=(224, 224),
#     batch_size=batch_size,
#     shuffle=False,
#     class_mode=None)
# activations = modified_model.predict_generator(
#     generator,
#     steps=ceil(generator.n/generator.batch_size),
#     use_multiprocessing=True,
#     workers=7,
#     verbose=True)
#
# mean_activations = []
# for i in range(generator.num_classes):
#     all_activations = activations[np.flatnonzero(generator.classes==i)]
#     mean_activations.append(np.mean(all_activations, axis=0))
#     np.save(
#         path_to_activations+f'class{i:04}_all_activations'
#         all_activations,
#         allow_pickle=False)
# mean_activations = np.array(mean_activations)
# similarity = cosine_similarity(mean_activations)
# np.save(path_to_activations+'mean_activations', mean_activations, allow_pickle=False)
# np.save(path_to_activations+'cosine_similarity', similarity, allow_pickle=False)
