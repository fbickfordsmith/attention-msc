"""
Test attention networks on ImageNet.
"""

gpu = input('GPU: ')
type_category_set = input('Category-set type in {diff, sem, sim, size}: ')
version_wnids = input('Version number (WNIDs): ')
version_weights = input('Version number (weights): ')
start = int(input('Start category set: '))
stop = int(input('Stop category set (inclusive): '))

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import csv
import numpy as np
import pandas as pd
from ..utils.paths import (path_category_sets, path_imagenet, path_init_model,
    path_results, path_weights)
from ..utils.models import build_model
from ..utils.testing import predict_model, evaluate_predictions

ind_attention = 19
model = build_model(train=False, attention_position=ind_attention)
model.save_weights(path_init_model)
path_cat_sets = (
    path_category_sets/f'{type_category_set}_v{version_wnids}_wnids.csv')
category_sets = [row for row in csv.reader(open(path_cat_sets), delimiter=',')]
scores_in, scores_out = [], []

for i in range(start, stop+1):
    name_weights = f'{type_category_set}_v{version_weights}_{i:02}'
    print(f'\nTesting on {name_weights}')
    weights = np.load(path_weights/f'{name_weights}_weights.npy')
    model.load_weights(path_init_model)
    model.layers[ind_attention].set_weights([weights])
    predictions, generator = predict_model(
        model, 'dir', path_imagenet/'val_white/')
    wnid2ind = generator.class_indices
    labels = generator.classes
    inds_in = []
    for wnid in category_sets[i]:
        inds_in.extend(np.flatnonzero(labels==wnid2ind[wnid]))
    inds_out = np.setdiff1d(range(generator.n), inds_in)
    print(f'''
        In category_set: {len(inds_in)} examples
        Out of category_set: {len(inds_out)} examples''')
    scores_in.append(evaluate_predictions(predictions, labels, inds_in))
    scores_out.append(evaluate_predictions(predictions, labels, inds_out))

cols_array = ['loss_in', 'acc_top1_in', 'acc_top5_in', 'loss_out',
    'acc_top1_out', 'acc_top5_out']
cols_save = ['loss_in', 'loss_out', 'acc_top1_in', 'acc_top1_out',
    'acc_top5_in', 'acc_top5_out']

scores_all = np.concatenate((np.array(scores_in), np.array(scores_out)), axis=1)
scores_df = pd.DataFrame(scores_all, columns=cols_array)
scores_df[cols_save].to_csv(
    (path_results/
    f'{type_category_set}_v{version_weights}_{start:02}-{stop:02}_results.csv'))
