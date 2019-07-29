'''
Define routines for finding a model's
- Predictions (outputs)
- Performance (metrics)
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

generator_params_test = dict(
    target_size=(224, 224),
    batch_size=256,
    shuffle=False)

def compute_steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def predict_model(model, path_data):
    generator = datagen_test.flow_from_directory(
        directory=path_data,
        class_mode=None, # None => returns just images (no labels)
        **generator_params_test)
    predictions = model.predict_generator(
        generator=generator,
        steps=compute_steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)
    return predictions, generator

def evaluate_model(model, path_data):
    generator = datagen_test.flow_from_directory(
        directory=path_data,
        class_mode='categorical',
        **generator_params_test)
    scores = model.evaluate_generator(
        generator=generator,
        steps=compute_steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)
    return scores
