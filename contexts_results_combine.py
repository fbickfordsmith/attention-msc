'''
Combine results that have been saved separately during model evaluation.
'''

import sys
import os
from glob import glob1
import numpy as np
import pandas as pd

_, type_context = sys.argv
# path_results = '/home/freddie/attention/results/'
path_results = 'results/'
num_contexts = int(0.5 * len(
    [file for file in glob1(path_results, '*.npy') if type_context in file]))

scores_ic, scores_ooc = [], []
for i in range(num_contexts):
    scores_ic.append(np.load(f'{path_results}{type_context}contexts_incontext{i:02}.npy'))
    scores_ooc.append(np.load(f'{path_results}{type_context}contexts_outofcontext{i:02}.npy'))
scores_arr = np.concatenate((np.array(scores_ic), np.array(scores_ooc)), axis=1)

col_names = [
    'incontext_loss', 'incontext_acc', 'incontext_top5acc',
    'outofcontext_loss', 'outofcontext_acc', 'outofcontext_top5acc']

pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'{path_results}{type_context}contexts_trained_metrics.csv')

# for file in glob1(path_results, '*.npy'):
#     if type_context in file:
#         os.remove(path_results+file)
