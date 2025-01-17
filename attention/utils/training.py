"""
Train a model using either `flow_from_directory` or `flow_from_dataframe`.
"""

import numpy as np
import pandas as pd
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from ..utils.metadata import wnids
from ..utils.paths import path_repo
from ..utils.preprocessing import crop_and_pca_generator

split = 0.1

datagen_valid = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=split)

early_stopping = EarlyStopping(
    min_delta=0.001,
    patience=2,
    verbose=True,
    restore_best_weights=True)

params_training = dict(
    epochs=300,
    verbose=1,
    callbacks=[early_stopping],
    use_multiprocessing=False,
    workers=1)

params_generator = dict(
    batch_size=256,
    shuffle=True,
    class_mode='categorical')

def partition_shuffled(df, labels_col='class'):
    df_train, df_valid = train_test_split(df, test_size=split, stratify=df[labels_col])
    return pd.concat((df_valid, df_train))

def partition_ordered(df, labels_col='class'):
    df_train, df_valid = pd.DataFrame(), pd.DataFrame()
    for wnid in wnids:
        inds = np.flatnonzero(df[labels_col]==wnid)
        val_size = int(split*len(inds))
        df_train = df_train.append(df.iloc[inds[val_size:]])
        df_valid = df_valid.append(df.iloc[inds[:val_size]])
    return pd.concat((df_valid, df_train))

def train_model(model, type_source, *args, use_data_aug=False):
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
    if type_source == 'dir':
        path_directory = args[0]
        generator_train = datagen_train.flow_from_directory(
            subset='training',
            target_size=target_size_train,
            directory=path_directory,
            **params_generator)
        generator_valid = datagen_train.flow_from_directory(
            subset='validation',
            target_size=(224,224),
            directory=path_directory,
            **params_generator)
    else:
        dataframe, path_data = args
        params_generator.update(dict(
            dataframe=partition_ordered(dataframe),
            directory=path_data,
            classes=wnids))
        generator_train = datagen_train.flow_from_dataframe(
            subset='training',
            target_size=target_size_train,
            **params_generator)
        generator_valid = datagen_valid.flow_from_dataframe(
            subset='validation',
            target_size=(224,224),
            **params_generator)
    params_training.update(dict(
        steps_per_epoch=len(generator_train),
        validation_data=generator_valid,
        validation_steps=len(generator_valid)))
    if use_data_aug:
        history = model.fit_generator(
            generator=crop_and_pca_generator(generator_train, crop_length=224),
            **params_training)
    else:
        history = model.fit_generator(
            generator=generator_train,
            **params_training)
    return model, history
