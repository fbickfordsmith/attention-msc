'''
ImageNet classes have been grouped into contexts. For each context, an
attention model has been trained on examples from that context only. For each
trained model, evaluate on val_white examples.

Command-line arguments:
1. type_context in {diff, sim, sem, size}
2. context_start in [0, num_contexts-1]
3. context_end in [context_start+1, num_contexts]
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import sys
import itertools
import numpy as np
import pandas as pd
from keras.layers import Lambda
from models import build_model
from testing import evaluate_model

_, type_context, context_start, context_end = sys.argv
path_weights = '/home/freddie/attention/weights/'
path_data = f'/home/freddie/ILSVRC2012-{type_context}contexts/val_white/'
num_contexts = len(os.listdir(path_data))
scores_incontext, scores_outofcontext = [], []

for i in range(int(context_start), int(context_end)):
# for i in range(num_contexts):
    print(f'\nEvaluating model trained on {type_context}context {i}')
    W = np.load(f'{path_weights}{type_context}context{i:02}_attention_weights.npy')
    model = build_model(Lambda(lambda x: W * x), train=False)
    # model = build_model(Attention(), train=False)
    # model.layers[19].set_weights([W])

    # evaluate on in-context data
    scores_ic = np.array(evaluate_model(model, f'{path_data}context{i:02}'))
    scores_incontext.append(scores_ic)
#    np.save(
#        f'results/{type_context}contexts_incontext{i:02}.npy',
#        scores_ic,
#        allow_pickle=False)

    # evaluate on out-of-context data
    scores_ooc = []
    for j in range(num_contexts):
        if j != i:
            scores_ooc.append(
                evaluate_model(model, f'{path_data}context{j:02}'))
    scores_ooc = np.mean(np.array(scores_ooc), axis=0)
    scores_outofcontext.append(scores_ooc)
#    np.save(
#        f'results/{type_context}contexts_outofcontext{i:02}.npy',
#        scores_ooc,
#        allow_pickle=False)

scores_arr = np.concatenate((
    np.array(scores_incontext),
    np.array(scores_outofcontext)),
    axis=1)

# list(itertools.chain(list1, list2, ...)) returns a flattened list
col_names = list(itertools.chain(
    [f'incontext_{metric_name}' for metric_name in model.metrics_names],
    [f'outofcontext_{metric_name}' for metric_name in model.metrics_names]))

pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'results/{type_context}contexts_trained_metrics.csv')
