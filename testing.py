'''
Define a routine for evaluating the performance of a model.

Define a routine for predicting
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

def evaluate_model(model, path_data):
    generator = datagen_test.flow_from_directory(
        directory=path_data,
        class_mode='categorical',
        **generator_params_test)

    score = model.evaluate_generator(
        generator=generator,
        steps=compute_steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)

    return score

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

    # probabilities = model.predict_generator(
    #     generator=generator,
    #     steps=compute_steps(generator.n, generator.batch_size),
    #     use_multiprocessing=False,
    #     verbose=True)
    #
    # classes_pred = np.argmax(probabilities, axis=1)
    # classes_true = generator.classes
    # wnid2ind = generator.class_indices
    #
    # return probabilities, classes_pred, classes_true, wnid2ind
