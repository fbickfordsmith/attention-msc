'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples.

Command-line arguments:
1. type_context in {diff, sim, sem, size}
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import csv
import itertools
import numpy as np
import pandas as pd
from layers import Attention
from models import build_model
from testing import predict_model

_, type_context = sys.argv
data_partition = 'val_white'
type_source = 'directory'
path_weights = '/home/freddie/attention/weights/'
path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_splitdata = f'/home/freddie/ILSVRC2012-{type_context}contexts/{data_partition}/'
path_dataframes = f'/home/freddie/dataframes_{data_partition}/{type_context}contexts/'
path_initmodel = f'/home/freddie/keras-models/{type_context}contexts_initialised_model.h5'
path_contexts = f'/home/freddie/attention/contexts/{type_context}contexts_wnids.csv'
path_results = '/home/freddie/attention/results/'
model = build_model(Attention(), train=False, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]
contexts = [row for row in csv.reader(open(path_contexts), delimiter=',')]
scores_ic, scores_ooc = [], []

for i, context in enumerate(contexts):
    name_context = f'{type_context}context{i:02}'
    print(f'\nTesting on {name_context}')
    W = np.load(f'{path_weights}{name_context}_weights_v5.npy')
    # W = np.load(f'{path_weights}{name_context}_weights.npy')
    model.load_weights(path_initmodel) # `del model` deletes an existing model
    model.layers[ind_attention].set_weights([W])
    predictions, generator = predict_model(model, type_source, path_data)
    wnid2ind = generator.class_indices
    labels = generator.classes

    inds_incontext = []
    for wnid in context:
        inds_incontext.extend(np.flatnonzero(labels==wnid2ind[wnid]))
    inds_outofcontext = np.setdiff1d(range(generator.n), inds_incontext)

    scores_ic.append(evaluate_predictions(predictions, labels, inds_incontext))
    scores_ooc.append(evaluate_predictions(predictions, labels, inds_outofcontext))

col_names = []
col_names.extend([f'incontext_{name}' for name in model.metrics_names])
col_names.extend([f'outofcontext_{name}' for name in model.metrics_names])

pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'{path_results}{type_context}contexts_trained_metrics.csv')