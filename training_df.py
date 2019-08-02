'''
Define a routine for training a model using flow_from_dataframe.
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from img_processing import crop_and_pca_generator

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'
wnids = [line.rstrip('\n') for line in open(path_synsets)]

datagen_train = ImageDataGenerator(
    fill_mode='nearest',
    horizontal_flip=True,
    rescale=None,
    data_format='channels_last',
    preprocessing_function=preprocess_input,
    validation_split=0.1)

datagen_valid = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1)

params_generator = dict(
    directory=path_data,
    batch_size=256,
    shuffle=True,
    class_mode='categorical',
    x_col='path',
    y_col='wnid',
    classes=wnids)

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
    workers=7)

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def train_model(model, dataframe):
    train_generator = datagen_train.flow_from_dataframe(
        dataframe=dataframe,
        subset='training',
        target_size=(256, 256),
        **params_generator)

    valid_generator = datagen_valid.flow_from_dataframe(
        dataframe=dataframe,
        subset='validation',
        target_size=(224, 224),
        **params_generator)

    train_generator_aug = crop_and_pca_generator(train_generator, crop_length=224)

    history = model.fit_generator(
        generator=train_generator_aug,
        steps_per_epoch=steps(train_generator.n, train_generator.batch_size),
        validation_data=valid_generator,
        validation_steps=steps(valid_generator.n, valid_generator.batch_size),
        **params_training)

    return model, history
