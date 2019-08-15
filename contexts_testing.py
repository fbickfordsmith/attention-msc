'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples. Runtime: ~3 mins/context.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import csv
import numpy as np
import pandas as pd
from models import build_model
from testing import predict_model, evaluate_predictions

type_context = input('Context type in {diff, sem, sim, size}: ')
version_wnids = 'v' + input('Version number (WNIDs): ')
version_weights = 'v' + input('Version number (weights): ')
start = int(input('Start context: '))
stop = int(input('Stop context (inclusive): '))
data_partition = 'val_white'

path_weights = '/home/freddie/attention/weights/'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_initmodel = '/home/freddie/keras-models/initialised_model.h5'
path_contexts = f'/home/freddie/attention/contexts/{type_context}contexts_wnids_{version_wnids}.csv'
path_results = '/home/freddie/attention/results/'

model = build_model(train=False, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]
contexts = [row for row in csv.reader(open(path_contexts), delimiter=',')]
scores_in, scores_out = [], []

for i in range(start, stop+1):
    name_weights = f'{type_context}_{version_weights}'
    print(f'\nTesting on {name_weights}')
    weights = np.load(f'{path_weights}{name_weights}_weights.npy')
    model.load_weights(path_initmodel) # `del model` deletes `model`
    model.layers[ind_attention].set_weights([weights])
    predictions, generator = predict_model(model, 'directory', path_data)
    wnid2ind = generator.class_indices
    labels = generator.classes
    inds_in = []
    for wnid in contexts[i]:
        inds_in.extend(np.flatnonzero(labels==wnid2ind[wnid]))
    inds_out = np.setdiff1d(range(generator.n), inds_in)
    print(f'''
        In context: {len(inds_in)} examples
        Out of context: {len(inds_out)} examples''')
    scores_in.append(evaluate_predictions(predictions, labels, inds_in))
    scores_out.append(evaluate_predictions(predictions, labels, inds_out))

col_names = ['loss_in', 'loss_out', 'acc_top1_in', 'acc_top1_out', 'acc_top5_in', 'acc_top5_out']
scores_all = np.concatenate((np.array(scores_in), np.array(scores_out)), axis=1)
pd.DataFrame(scores_all, columns=col_names).to_csv(
    f'{path_results}{name_weights}_{start}-{stop}_results.csv')
