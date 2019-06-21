'''
ImageNet classes have been grouped by baseline accuracy into 20 sets. For each
set, train an attention layer on examples from that set only.

References:
- stackoverflow.com/questions/43906048/keras-early-stopping
- stackoverflow.com/questions/40496069/reset-weights-in-keras-layer/50257383
- stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from attention_model import build_model, train_model

path_to_weights = '/home/freddie/keras-models/'
path_to_data = '/home/freddie/ILSVRC2012/clsloc/train/'
batch_size = 256 # VGG paper

datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1)

generator_params = dict(
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True, # False => returns images in order
    class_mode='categorical') # None => returns just images (no labels)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=1, # number of epochs without improvement after which we stop
    verbose=True,
    restore_best_weights=True) # False => weights from last step are used

training_params = dict(
    epochs=15,
    verbose=1,
    callbacks=[early_stopping],
    use_multiprocessing=True,
    workers=7)

attention_model = build_model()
attention_model.save_weights(path_to_weights+'initialised_model.h5')
class_sets = np.loadtxt('csv/class_sets.csv', dtype=str, delimiter=',')

for i, class_set in enumerate(class_sets):
    print(f'Training on class set {i}')
    path_to_set = path_to_data + f'set{i:02}'
    attention_model.load_weights(path_to_weights+'initialised_model.h5')
    attention_model, history = train_model(
        model=attention_model,
        datagen=datagen,
        datapath=path_to_set,
        generator_params=generator_params,
        training_params=training_params)
    model_name = f'set{i:02}_model'
    pd.DataFrame(history.history).to_csv('csv/'+model_name+'_results.csv')
    np.save(
        path_to_weights+model_name+'_attention_weights',
        attention_model.layers[19].get_weights()[0],
        allow_pickle=False)
    attention_model.save_weights(path_to_weights+model_name+'_all_weights.h5')
