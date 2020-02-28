"""
Define paths used throughout the repository.
"""

import sys
import pathlib

if sys.platform == 'linux':
    path_repo = pathlib.Path('/home/freddie/attention/')
    path_imagenet = pathlib.Path('/fast-data/datasets/ILSVRC/2012/clsloc/')
    path_dataframes = pathlib.Path('/home/freddie/dataframes/')
    path_init_model = pathlib.Path('/home/freddie/initialised_model.h5')
    path_activations = pathlib.Path('/home/freddie/activations/')
else:
    path_repo = pathlib.Path('/Users/fbickfordsmith/Google Drive/Project/attention/')

path_category_sets = path_repo/'data/category_sets/'
path_metadata = path_repo/'data/metadata/'
path_representations = path_repo/'data/representations/'
path_results = path_repo/'data/results/'
path_training = path_repo/'data/training/'
path_weights = path_repo/'data/weights/'
