'''
ImageNet classes have been grouped by baseline accuracy into 20 'contexts'.

For each context, an attention model has been trained on examples from that
context only.

For each trained model, evaluate on val_white data.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import itertools
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from attention_model import build_model

path_weights = '/home/freddie/keras-models/'
path_separated_data = '/home/freddie/ILSVRC2012/clsloc/val_white/'
path_all_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/'
batch_size = 256
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
model = build_model()
incontext_scores, outofcontext_scores, alldata_scores = [], [], []

def evaluate_by_path(model, path_data):
    test_generator = datagen.flow_from_directory(
        directory=path_data,
        target_size=(224, 224),
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')

    score = model.evaluate_generator(
        test_generator,
        steps=int(np.ceil(test_generator.n/test_generator.batch_size)),
        use_multiprocessing=False,
        verbose=True)

    return score

for i in range(20):
    print(f'\nEvaluating model trained on set {i}')
    model.load_weights(path_weights+f'set{i:02}_model_all_weights.h5')

    # evaluate on in-set data
    incontext_scores.append(
        evaluate_by_path(model, path_separated_data+f'set{i:02}'))

    # evaluate on out-of-set data
    scores_temp = [
        evaluate_by_path(model, path_separated_data+f'set{j:02}')
        for j in (set(range(20)) - set([i]))]
    outofcontext_scores.append(np.mean(np.array(scores_temp), axis=0))

    # evaluate on all data
    alldata_scores.append(
        evaluate_by_path(model, path_all_data))

scores_arr = np.concatenate((
    np.array(incontext_scores),
    np.array(outofcontext_scores),
    np.array(alldata_scores)),
    axis=1)

# list(itertools.chain(list1, list2, ...)) returns a flattened list
col_names = list(itertools.chain(
    ['incontext_'+metric_name for metric_name in model.metrics_names],
    ['outofcontext_'+metric_name for metric_name in model.metrics_names],
    ['alldata_'+metric_name for metric_name in model.metrics_names]))

scores_df = pd.DataFrame(scores_arr, columns=col_names)
scores_df.to_csv('csv/test_diff_contexts_results.csv')
