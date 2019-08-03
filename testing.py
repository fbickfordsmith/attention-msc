'''
Define a routine for finding a model's predictions and performance using either
flow_from_directory or flow_from_dataframe.
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

path_synsets = '/home/freddie/attention/metadata/synsets.txt'
wnids = [line.rstrip('\n') for line in open(path_synsets)]

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

params_generator = dict(
    target_size=(224, 224),
    batch_size=256,
    shuffle=False)

params_testing = dict(
    use_multiprocessing=False,
    verbose=True)

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def evaluate_model(model, type_source, *args):
    if type_source == 'directory':
        path_directory = args[0]
        generator = datagen.flow_from_directory(
            directory=path_directory,
            class_mode='categorical',
            **params_generator)
    else:
        dataframe, path_data = args
        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=path_data,
            class_mode='categorical',
            classes=wnids,
            **params_generator)

    scores = model.evaluate_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        **params_testing)

    return scores

def predict_model(model, type_source, *args):
    if type_source == 'directory':
        path_directory = args[0]
        generator = datagen.flow_from_directory(
            directory=path_directory,
            class_mode=None, # None => returns just images (no labels)
            **params_generator)
    else:
        dataframe, path_data = args
        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=path_data,
            class_mode=None,
            **params_generator)

    predictions = model.predict_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        **params_testing)

    return predictions, generator
