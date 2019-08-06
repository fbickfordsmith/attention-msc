import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from models import build_model
from keras.applications.vgg16 import preprocess_input
from models_vgg2 import build_vgg2
import time

data_partition = 'train'
path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_activations = '/home/freddie/activations-conv/'
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

generator0 = datagen.flow_from_directory(
    directory=path_data,
    class_mode=None, # None => returns just images (no labels)
    target_size=(224, 224),
    batch_size=256,
    shuffle=True)

generator1 = datagen.flow_from_directory(
    directory=path_activations,
    class_mode=None, # None => returns just images (no labels)
    target_size=(224, 224),
    batch_size=256,
    shuffle=True)

time0 = time.time()
model0 = build_model()
print(f'Old model: build time = {time.time()-time0} seconds')

time0 = time.time()
model1 = build_vgg2()
print(f'New model: build time = {time.time()-time0} seconds')

time0 = time.time()
predictions0 = model0.predict_generator(
    generator=generator0,
    steps=steps(generator.n, generator.batch_size),
    use_multiprocessing=True,
    verbose=True)
print(f'Old model: epoch time = {time.time()-time0} seconds')

time0 = time.time()
predictions1 = model1.predict_generator(
    generator=generator,
    steps=steps(generator.n, generator.batch_size),
    use_multiprocessing=True,
    verbose=True)
print(f'New model: epoch time = {time.time()-time0} seconds')
