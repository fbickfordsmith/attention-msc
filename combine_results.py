import os
import pandas as pd

type_context = input('Context type in {diff, sem, sim, size}: ')
version_weights = 'v' + input('Version number (weights): ')

path_results = '/Users/fbickfordsmith/Google Drive/Project/attention/results/'
id_context = f'{type_context}_{version_weights}'
filenames = sorted([f for f in os.listdir(path_results) if id_context in f])

df = pd.concat(
    [pd.read_csv(path_results+f, index_col=0) for f in filenames],
    ignore_index=True)

df.to_csv(f'{path_results}{id_context}_results.csv')