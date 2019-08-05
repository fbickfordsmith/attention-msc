'''
Define routines for finding a model's predictions and performance using either
flow_from_directory or flow_from_dataframe. Also define a routine for
evaluating performance using predictions and labels.
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics, losses
from keras import backend as K

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

def evaluate_model(model, type_source, *args):
    if type_source == 'directory':
        path_directory = args[0]
        generator = datagen.flow_from_directory(
            directory=path_directory,
            class_mode='sparse',
            # class_mode='categorical',
            **params_generator)
    else:
        dataframe, path_data = args
        generator = datagen.flow_from_dataframe(
            dataframe=dataframe,
            directory=path_data,
            class_mode='sparse',
            # class_mode='categorical',
            classes=wnids,
            **params_generator)

    scores = model.evaluate_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        **params_testing)

    return scores

def evaluate_predictions(predictions, labels, indices):
    ypred = K.variable(predictions[indices])
    ytrue = K.variable(labels[indices])
    sess = K.get_session()
    acc_top1 = sess.run(metrics.sparse_top_k_categorical_accuracy(ytrue, ypred, k=1))
    acc_top5 = sess.run(metrics.sparse_top_k_categorical_accuracy(ytrue, ypred, k=5))
    loss = sess.run(K.mean(losses.sparse_categorical_crossentropy(ytrue, ypred)))
    return loss, acc_top1, acc_top5
