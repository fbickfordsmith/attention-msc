'''
References:
- stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
- stackoverflow.com/questions/43906048/keras-early-stopping
'''

import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1)

generator_params_train = dict(
    target_size=(224, 224),
    batch_size=256,
    shuffle=True,
    class_mode='categorical')

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=1, # number of epochs without improvement after which we stop
    verbose=True,
    restore_best_weights=True) # False => weights from last step are used

training_params = dict(
    epochs=1000,
    verbose=1,
    callbacks=[early_stopping],
    use_multiprocessing=True,
    workers=7)

def compute_steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def train_model(model, path_data):
    train_generator = datagen_train.flow_from_directory(
        directory=path_data,
        subset='training',
        **generator_params_train)

    valid_generator = datagen_train.flow_from_directory(
        directory=path_data,
        subset='validation',
        **generator_params_train)

    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=compute_steps(
            train_generator.n, train_generator.batch_size),
        validation_data=valid_generator,
        validation_steps=compute_steps(
            valid_generator.n, valid_generator.batch_size),
        **training_params)

    return model, history
