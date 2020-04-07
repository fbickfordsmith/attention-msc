"""
Combine results files generated by `attention_networks_testing.py` on separate
GPUs.
"""

type_category_set = input('Category-set type in {diff, sem, sim, size}: ')
version_weights = input('Version number (weights): ')
id_category_set = f'{type_category_set}_v{version_weights}'

import os
import pandas as pd
from ..utils.paths import path_results

filenames = sorted([f for f in os.listdir(path_results) if id_category_set in f])

df = pd.concat(
    [pd.read_csv(path_results/f, index_col=0) for f in filenames],
    ignore_index=True)

df.to_csv(path_results/f'{id_category_set}_results.csv')
