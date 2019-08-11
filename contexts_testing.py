'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples. Runtime: ~3 mins/context.

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
from models import build_model
from testing import predict_model, evaluate_predictions

_, type_context = sys.argv
data_partition = 'val_white'
path_weights = '/home/freddie/attention/weights/'
path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_initmodel = f'/home/freddie/keras-models/{type_context}contexts_initialised_model.h5'
path_contexts = f'/home/freddie/attention/contexts/{type_context}contexts_wnids.csv'
path_results = '/home/freddie/attention/results/'
model = build_model(train=False, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]
contexts = [row for row in csv.reader(open(path_contexts), delimiter=',')]
scores_ic, scores_ooc = [], []

start, stop = 18, 25

#for i, context in enumerate(contexts[start:stop]):
for i in range(start, stop):
    name_context = f'{type_context}context{i:02}'
    print(f'\nTesting on {name_context}')
    weights = np.load(f'{path_weights}{name_context}_weights.npy')
    model.load_weights(path_initmodel) # `del model` deletes an existing model
    model.layers[ind_attention].set_weights([weights])
    predictions, generator = predict_model(model, 'directory', path_data)
    wnid2ind = generator.class_indices
    labels = generator.classes
    inds_incontext = []
    for wnid in contexts[i]:
        inds_incontext.extend(np.flatnonzero(labels==wnid2ind[wnid]))
    inds_outofcontext = np.setdiff1d(range(generator.n), inds_incontext)
    print(f'''
        In context: {len(inds_incontext)} examples
        Out of context: {len(inds_outofcontext)} examples''')
    scores_ic.append(evaluate_predictions(predictions, labels, inds_incontext))
    scores_ooc.append(evaluate_predictions(predictions, labels, inds_outofcontext))

col_names = []
col_names.extend([f'incontext_{m}' for m in ['loss', 'acc_top1', 'acc_top5']])
col_names.extend([f'outofcontext_{m}' for m in ['loss', 'acc_top1', 'acc_top5']])
scores_arr = np.concatenate((np.array(scores_ic), np.array(scores_ooc)), axis=1)
pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'{path_results}{type_context}contexts_trained_metrics_{start}{stop}.csv')
