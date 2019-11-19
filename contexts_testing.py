'''
Evaluate trained attention networks on validation-set ImageNet examples.
Runtime: ~3 mins/context.
'''

gpu = input('GPU: ')
type_context = input('Context type in {diff, sem, sim, size}: ')
version_wnids = 'v' + input('Version number (WNIDs): ')
version_weights = 'v' + input('Version number (weights): ')
start = int(input('Start context: '))
stop = int(input('Stop context (inclusive): '))

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import csv
import numpy as np
import pandas as pd
from models import build_model
from testing import predict_model, evaluate_predictions

data_partition = 'val_white'

path_weights = '/home/freddie/attention/weights/'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_initmodel = f'/home/freddie/initialised_model_{start:02}-{stop:02}.h5'
path_contexts = f'/home/freddie/attention/contexts/{type_context}_{version_wnids}_wnids.csv'
path_results = '/home/freddie/attention/results/'

model = build_model(train=False, attention_position=19)
model.save_weights(path_initmodel)
ind_attention = np.flatnonzero(['attention' in layer.name for layer in model.layers])[0]
contexts = [row for row in csv.reader(open(path_contexts), delimiter=',')]
scores_in, scores_out = [], []

for i in range(start, stop+1):
    name_weights = f'{type_context}_{version_weights}_{i:02}'
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

cols_array = ['loss_in', 'acc_top1_in', 'acc_top5_in', 'loss_out', 'acc_top1_out', 'acc_top5_out']
cols_save = ['loss_in', 'loss_out', 'acc_top1_in', 'acc_top1_out', 'acc_top5_in', 'acc_top5_out']

scores_all = np.concatenate((np.array(scores_in), np.array(scores_out)), axis=1)
scores_df = pd.DataFrame(scores_all, columns=cols_array)
scores_df[cols_save].to_csv(
    f'{path_results}{type_context}_{version_weights}_{start:02}-{stop:02}_results.csv')
