"""
Define metadata variables used throughout the repository.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_distances
from ..utils.paths import path_metadata, path_representations, path_results

wnids = open(path_metadata/'imagenet_class_wnids.txt').read().splitlines()
ind2wnid = {ind:wnid for ind, wnid in enumerate(wnids)}
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}

df_baseline = pd.read_csv(
    path_results/'baseline_vgg16_results.csv', index_col=0)
mean_acc = np.mean(df_baseline['accuracy'])
std_acc = np.std(df_baseline['accuracy'])

representations = np.load(path_representations/'representations_mean.npy')
represent_dist = cosine_distances(representations)
mean_dist = np.mean(squareform(represent_dist, checks=False))
std_dist = np.std(squareform(represent_dist, checks=False))
