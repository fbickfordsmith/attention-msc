'''
Define routines for finding a model's predictions and performance using
flow_from_directory.
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

params_generator = dict(
    target_size=(224, 224),
    batch_size=256,
    shuffle=False)

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def predict_model(model, path_data):
    generator = datagen_test.flow_from_directory(
        directory=path_data,
        class_mode=None, # None => returns just images (no labels)
        **params_generator)
    predictions = model.predict_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)
    return predictions, generator

def evaluate_model(model, path_data):
    generator = datagen_test.flow_from_directory(
        directory=path_data,
        class_mode='categorical',
        **params_generator)
    scores = model.evaluate_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)
    return scores
