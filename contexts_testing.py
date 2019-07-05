'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples.

Command-line arguments:
1. type_context in {diff, sim}.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import itertools
import numpy as np
import pandas as pd
from layers import FixedWeightAttention
from models import build_model
from testing import evaluate_model

_, type_context = sys.argv
path_weights = '/home/freddie/keras-models/'
path_splitdata = f'/home/freddie/ILSVRC2012-{type_context}contexts/val_white/'
path_alldata = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white/'
num_contexts = len(os.listdir(path_splitdata))
scores_incontext, scores_outofcontext, scores_alldata = [], [], []

for i in range(num_contexts):
    print(f'\nEvaluating model trained on {type_context}context {i}')
    W = np.load(path_weights+f'{type_context}context{i:02}_attention_weights.npy')
    model = build_model(FixedWeightAttention(W), train=False)

    # evaluate on in-context data
    scores_incontext.append(
        evaluate_model(model, path_splitdata+f'context{i:02}'))

    # evaluate on out-of-context data
    scores_temp = np.array([
        evaluate_model(model, path_splitdata+f'context{j:02}')
        for j in range(num_contexts) if j != i])
    scores_outofcontext.append(np.mean(scores_temp, axis=0))

    # evaluate on all data
    scores_alldata.append(evaluate_model(model, path_alldata))

scores_arr = np.concatenate((
    np.array(scores_incontext),
    np.array(scores_outofcontext),
    np.array(scores_alldata)),
    axis=1)

# list(itertools.chain(list1, list2, ...)) returns a flattened list
col_names = list(itertools.chain(
    [f'incontext_{metric_name}' for metric_name in model.metrics_names],
    [f'outofcontext_{metric_name}' for metric_name in model.metrics_names],
    [f'all_data_{metric_name}' for metric_name in model.metrics_names]))

pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'results/{type_context}contexts_trained_metrics.csv')
