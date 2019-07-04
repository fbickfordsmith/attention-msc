import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from attention_model import build_model, traicdn_model

path_weights = '~/keras-models/'
path_data = '~/ILSVRC2012-simcontexts/train/'
batch_size = 256

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
attention_model.save_weights(path_weights+'initialised_model_simcontexts.h5')

for i in range(40):
    print(f'Training on context {i}')
    path_context = path_data + f'context{i:02}'
    attention_model.load_weights(path_weights+'initialised_model_simcontexts.h5')
    attention_model, history = train_model(
        model=attention_model,
        datagen=datagen,
        datapath=path_context,
        generator_params=generator_params,
        training_params=training_params)
    model_name = f'simcontext{i:02}'
    pd.DataFrame(history.history).to_csv('csv/'+model_name+'_training.csv')
    np.save(
        path_weights+model_name+'_attention_weights',
        attention_model.layers[19].get_weights()[0],
        allow_pickle=False)
    # attention_model.save_weights(path_weights+model_name+'_all_weights.h5')
