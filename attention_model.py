'''
Define a model with elementwise-multiplication attention layer as the only
trainable layer.

Define a routine for training one using a data generator.

References:
- stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
'''

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Input
from keras import optimizers
from attention_layer import Attention

def build_model(optimizer=optimizers.Adam(lr=3e-4)):
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
    print()

    attention_model.compile(
        optimizer=optimizer,
        # optimizer=optimizers.Adam(lr=3e-4), # Karpathy default
        # optimizer=optimizers.SGD(lr=1e-3, momentum=0.9), # could also try 1e-4
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']) # top-1 and top5 acc

    return attention_model

def train_model(model, datagen, datapath, generator_params, training_params):
    '''
    Examples:
    generator_params = dict(
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    training_params = dict(
        epochs=15,
        verbose=1,
        callbacks=[early_stopping],
        use_multiprocessing=True,
        workers=7)
    '''

    train_generator = datagen.flow_from_directory(
        **generator_params,
        directory=datapath,
        subset='training')

    valid_generator = datagen.flow_from_directory(
        **generator_params,
        directory=datapath,
        subset='validation')

    history = model.fit_generator(
        **training_params,
        generator=train_generator,
        steps_per_epoch=train_generator.n//train_generator.batch_size,
        validation_data=valid_generator,
        validation_steps=valid_generator.n//valid_generator.batch_size)

    return model, history
