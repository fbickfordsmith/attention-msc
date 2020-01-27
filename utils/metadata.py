"""
Define metadata variables used throughout the repository.
"""

import numpy as np
import pandas as pd
from .paths import path_repo

path_synsets = path_repo/'data/metadata/synsets.txt'
path_baseline = path_repo/'data/results/base_results.csv'
path_representations = path_repo/'data/representations/representations_mean.npy'

wnids = open(path_synsets).read().splitlines()
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

df_baseline = pd.read_csv(path_baseline, index_col=0)
mean_acc = np.mean(df_baseline['accuracy'])
std_acc = np.std(df_baseline['accuracy'])
lb_acc = mean_acc - std_acc
ub_acc = mean_acc + std_acc
inds_av_acc = np.flatnonzero(
    (lb_acc<df_baseline['accuracy']) & (df_baseline['accuracy']<ub_acc))

Z = np.load(path_representations)
