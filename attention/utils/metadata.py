"""
Define metadata variables used throughout the repository.
"""

import numpy as np
import pandas as pd
from ..utils.paths import path_metadata, path_representations, path_results

wnids = open(path_metadata/'synsets.txt').read().splitlines()
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}
df_baseline = pd.read_csv(path_results/'base_results.csv', index_col=0)
mean_acc = np.mean(df_baseline['accuracy'])
std_acc = np.std(df_baseline['accuracy'])
lb_acc = mean_acc - std_acc
ub_acc = mean_acc + std_acc
inds_av_acc = np.flatnonzero(
    (lb_acc<df_baseline['accuracy']) & (df_baseline['accuracy']<ub_acc))
Z = np.load(path_representations/'representations_mean.npy')
