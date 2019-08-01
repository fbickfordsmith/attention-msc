import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

path_synsets = '/home/freddie/attention/metadata/synsets.txt'
wnids = [line.rstrip('\n') for line in open(path_synsets)]

datagen_test = ImageDataGenerator(preprocessing_function=preprocess_input)

generator_params_test = dict(
    target_size=(224, 224),
    batch_size=256,
    shuffle=False,
    x_col='path',
    y_col='wnid',
    classes=wnids)

def steps(num_examples, batch_size):
    return int(np.ceil(num_examples/batch_size))

def evaluate_model(model, dataframe, path_data):
    generator = datagen_test.flow_from_dataframe(
        dataframe=dataframe,
        directory=path_data,
        class_mode='categorical',
        **generator_params_test)

    scores = model.evaluate_generator(
        generator=generator,
        steps=steps(generator.n, generator.batch_size),
        use_multiprocessing=False,
        verbose=True)

    return scores
