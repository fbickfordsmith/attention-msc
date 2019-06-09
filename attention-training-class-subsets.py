import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Take a pretrained VGG16.
Add an attention layer between the final conv layer and the first FC layer.
Fix all parameters except for the attention weights.
Train on ImageNet.

References:
github.com/keras-team/keras/blob/master/keras/constraints.py
stackoverflow.com/questions/46821845/how-to-add-a-trainable-hadamard-product-layer-in-keras
stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
stackoverflow.com/questions/43906048/keras-early-stopping
stackoverflow.com/questions/40496069/reset-weights-in-keras-layer/50257383
'''

import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from keras.engine.topology import Layer
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.metrics import sparse_top_k_categorical_accuracy
from attention_layer import Attention

################################################################################

pretrained_model = VGG16(weights='imagenet')
model_in = Input(batch_shape=(None, 224, 224, 3))
model_out = pretrained_model.layers[1](model_in)
for layer in pretrained_model.layers[2:19]:
    model_out = layer(model_out)
model_out = Attention()(model_out)
for layer in pretrained_model.layers[19:]:
    model_out = layer(model_out)
attention_model = Model(model_in, model_out)
for i, layer in enumerate(attention_model.layers):
    if i != 19:
        layer.trainable=False

print('\nAttention model layers:')
for i in list(enumerate(attention_model.layers)):
    print(i)
print('\nTrainable weights:')
for i in attention_model.trainable_weights:
    print(i)
print('\n')

attention_model.compile(
    optimizer=optimizers.Adam(lr=3e-4), # Karpathy default
    # optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), # relatively low lr (could also try 1e-4)
    loss='categorical_crossentropy',
    metrics=['accuracy', 'top_k_categorical_accuracy']) # top-1 and top5 acc
attention_model.save_weights(
    '/home/freddie/keras-models/initialised_attention_model.h5')

################################################################################

path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/'
batch_size = 256 # VGG paper
class_set_names = np.loadtxt('class-set-names.csv', dtype=str, delimiter=',')
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.1)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=1, # number of epochs without improvement after which we stop
    verbose=True,
    restore_best_weights=True) # False => weights from last step are used

for i, class_set in enumerate(class_set_names):
    print(f'Training on class set {i}')
    training_classes = list(class_set)
    train_generator = datagen.flow_from_directory(
        directory=path+'train/',
        classes=training_classes,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True, # False => returns images in order
        class_mode='categorical', # None => returns just images (no labels)
        subset='training')
    validation_generator = datagen.flow_from_directory(
        directory=path+'train/',
        classes=training_classes,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True, # False => returns images in order
        class_mode='categorical', # None => returns just images (no labels)
        subset='validation')
    attention_model.load_weights(
        '/home/freddie/keras-models/initialised_attention_model.h5')
    history = attention_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_generator.n//batch_size,
        epochs=15,
        verbose=1,
        callbacks=[early_stopping],
        validation_data=validation_generator,
        validation_steps=validation_generator.n//batch_size,
        use_multiprocessing=True,
        workers=7)
    model_name = f'attention-model-set{i}'
    results = pd.DataFrame(history.history)
    results.to_csv(model_name+'-results.csv')
    weights_to_save = attention_model.layers[19].get_weights()[0]
    np.save(model_name+'-attention-weights', weights_to_save, allow_pickle=False)
    attention_model.save_weights(
        '/home/freddie/keras-models/'+model_name+'-all-weights.h5')
