'''
Define a routine for evaluating a model using flow_from_dataframe.
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'
wnids = [line.rstrip('\n') for line in open(path_synsets)]

datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

params_generator = dict(
    directory=path_data,
    target_size=(224, 224),
    batch_size=256,
    shuffle=False,
    x_col='path',
    y_col='wnid',
    classes=wnids)

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def evaluate_model(model, dataframe):
    generator = datagen_test.flow_from_dataframe(
        dataframe=dataframe,
        class_mode='categorical',
        **params_generator)

    scores = model.evaluate_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)

    return scores
