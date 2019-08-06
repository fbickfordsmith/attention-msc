'''
Define a routine for training a model using either flow_from_directory or
flow_from_dataframe.

References:
- stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
- stackoverflow.com/questions/43906048/keras-early-stopping
'''

import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from img_processing import crop_and_pca_generator

path_synsets = '/home/freddie/attention/metadata/synsets.txt'
wnids = [line.rstrip('\n') for line in open(path_synsets)]
split = 0.1

datagen_valid = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=split)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=True,
    restore_best_weights=True)

params_generator = dict(
    batch_size=256,
    shuffle=True,
    class_mode='sparse')
    # class_mode='categorical')

params_training = dict(
    epochs=100,
    verbose=1,
    callbacks=[early_stopping],
    use_multiprocessing=True,
    workers=7)

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def stratified_shuffle(df, labels_col='class'):
    df_train, df_valid = train_test_split(df, test_size=split, stratify=df[labels_col])
    return pd.concat((df_valid, df_train))

def train_model(model, type_source, *args, use_data_aug=True):
    if use_data_aug:
        target_size_train = (256, 256)
        datagen_train = ImageDataGenerator(
            fill_mode='nearest',
            horizontal_flip=True,
            rescale=None,
            data_format='channels_last',
            preprocessing_function=preprocess_input,
            validation_split=split)
    else:
        target_size_train = (224, 224)
        datagen_train = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=split)

    if type_source == 'directory':
        path_directory = args[0]
        train_generator = datagen_train.flow_from_directory(
            subset='training',
            target_size=target_size_train,
            directory=path_directory,
            **params_generator)
        valid_generator = datagen_train.flow_from_directory(
            subset='validation',
            target_size=(224, 224),
            directory=path_directory,
            **params_generator)
    else:
        dataframe, path_data = args
        params_generator.update(dict(
            dataframe=stratified_shuffle(dataframe),
            directory=path_data,
            classes=wnids))
        train_generator = datagen_train.flow_from_dataframe(
            subset='training',
            target_size=target_size_train,
            **params_generator)
        valid_generator = datagen_valid.flow_from_dataframe(
            subset='validation',
            target_size=(224, 224),
            **params_generator)

    params_training.update(dict(
        steps_per_epoch=steps(train_generator.n, train_generator.batch_size),
        validation_data=valid_generator,
        validation_steps=steps(valid_generator.n, valid_generator.batch_size)))

    if use_data_aug:
        history = model.fit_generator(
            generator=crop_and_pca_generator(train_generator, crop_length=224),
            **params_training)
    else:
        history = model.fit_generator(
            generator=train_generator,
            **params_training)

    return model, history
