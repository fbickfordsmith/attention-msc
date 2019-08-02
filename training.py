'''
Define a routine for training a model using either flow_from_directory or
flow_from_dataframe.

References:
- stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
- stackoverflow.com/questions/43906048/keras-early-stopping
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from img_processing import crop_and_pca_generator

path_synsets = '/home/freddie/attention/metadata/synsets.txt'
# path_synsets = '/Users/fbickfordsmith/Google Drive/Project/attention/metadata/synsets.txt'
wnids = [line.rstrip('\n') for line in open(path_synsets)]

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def train_model(model, type_source, *args, use_data_aug=True):
    if type_source == 'directory':
        path_directory = args[0]
    else:
        dataframe, path_data = args

    if use_data_aug:
        datagen_train = ImageDataGenerator(
            fill_mode='nearest',
            horizontal_flip=True,
            rescale=None,
            data_format='channels_last',
            preprocessing_function=preprocess_input,
            validation_split=0.1)

    else:
        datagen_train = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.1)

    datagen_valid = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1)

    params_generator = dict(
        batch_size=256,
        shuffle=True,
        class_mode='categorical')

    if type_source == 'directory':
        train_generator = datagen_train.flow_from_directory(
            directory=path_directory,
            subset='training',
            target_size=(256, 256),
            **params_generator)

        valid_generator = datagen_train.flow_from_directory(
            directory=path_directory,
            subset='validation',
            target_size=(224, 224),
            **params_generator)

    else:
        params_dataframe = dict(
            dataframe=dataframe,
            directory=path_data,
            x_col='path',
            y_col='wnid',
            classes=wnids)

        train_generator = datagen_train.flow_from_dataframe(
            subset='training',
            target_size=(256, 256),
            **params_dataframe,
            **params_generator)

        valid_generator = datagen_valid.flow_from_dataframe(
            subset='validation',
            target_size=(224, 224),
            **params_dataframe,
            **params_generator)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        restore_best_weights=True)

    params_training = dict(
        epochs=100,
        verbose=1,
        callbacks=[early_stopping],
        use_multiprocessing=True,
        workers=7,
        steps_per_epoch=steps(train_generator.n, train_generator.batch_size),
        validation_data=valid_generator,
        validation_steps=steps(valid_generator.n, valid_generator.batch_size))

    if use_data_aug:
        train_generator_aug = crop_and_pca_generator(train_generator, crop_length=224)
        history = model.fit_generator(generator=train_generator_aug, **params_training)
    else:
        history = model.fit_generator(generator=train_generator, **params_training)

    return model, history
