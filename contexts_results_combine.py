import numpy as np
import pandas as pd

type_context = 'size' # 'sim'
num_contexts = 18 # 40

scores_incontext, scores_outofcontext = [], []
for i in range(num_contexts):
    scores_incontext.append(np.load(f'results/{type_context}contexts_incontext{i:02}.npy'))
    scores_outofcontext.append(np.load(f'results/{type_context}contexts_outofcontext{i:02}.npy'))
scores_arr = np.concatenate((np.array(scores_incontext), np.array(scores_outofcontext)), axis=1)

col_names = [
    'incontext_loss',
    'incontext_acc',
    'incontext_top5_acc',
    'outofcontext_loss',
    'outofcontext_acc',
    'outofcontext_top5_acc']

pd.DataFrame(scores_arr, columns=col_names).to_csv(
    f'results/{type_context}contexts_trained_metrics.csv')
