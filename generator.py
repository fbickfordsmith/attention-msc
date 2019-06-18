'''
Define a data generator based on keras.utils.Sequence. Define a routine for
instantiating generators.

References:
- keras.io/utils/
- stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
- medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4
- calebrob.com/ml/imagenet/ilsvrc2012/2018/10/22/imagenet-benchmarking.html
- github.com/HoldenCaulfieldRye/caffe/tree/master/data/ilsvrc12
'''

import os
import glob
import numpy as np
from keras import utils
from skimage.io import imread

class DataGenerator(utils.Sequence):
    def __init__(self, paths, labels, batch_size, preprocess_fn):
        self.paths = paths # list of paths to input files (eg, images)
        self.labels = labels # list of associated classes
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn # func: rgb_img -> processed_img

    def __len__(self):
        # return the number of batches to produce
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, ind):
        # return a complete batch
        batch_paths = self.paths[ind*self.batch_size:(ind+1)*self.batch_size]
        batch_inputs = [self.preprocess_fn(imread(p)) for p in batch_paths]
        batch_labels = self.labels[ind*self.batch_size:(ind+1)*self.batch_size]
        batch_onehots = utils.to_categorical(batch_labels, num_classes=1000)
        # return np.array(batch_inputs), np.array(batch_labels)
        return np.array(batch_inputs), batch_onehots

def listdir_names(path):
    return sorted([i for i in os.listdir(path) if not i.startswith('.')])

def listdir_paths(path):
    # path must have '/' at its end
    return sorted([i for i in glob.glob(path+'*') if not i.startswith('.')])

def get_filepaths_labels(path_data, path_synsets):
    # path_data in {'.../train/', '.../val/', '.../val_white/'}
    # path_synsets is like '.../synsets.txt'
    synsets_key = [line.rstrip('\n') for line in open(path_synsets)]
    syn2ind = {syn:ind for ind, syn in enumerate(synsets_key)}
    class_synsets = listdir_names(path_data)
    class_indices = [syn2ind[cs] for cs in class_synsets]
    filepaths, class_sizes = [], []
    for folderpath in listdir_paths(path_data):
        filepaths.extend(listdir_paths(folderpath+'/'))
        class_sizes.append(len(listdir_paths(folderpath+'/')))
    num_classes = len(class_synsets)
    labels = np.repeat(np.arange(num_classes), repeats=class_sizes)
    return filepaths, labels

def build_generators(path_data, path_synsets, batch_size, preprocess_fn, val_split=0):
    # path_data in {'.../train/', '.../val/', '.../val_white/'}
    filepaths_all, labels_all = get_filepaths_labels(path_data, path_synsets)

    if 0 < val_split < 1:
        # np.setdiff1d(a, b) returns unique values in a that are not in b
        num_examples = len(labels_all)
        num_train = int(np.ceil((1-val_split) * num_examples))
        ind_train = np.random.randint(num_examples, size=num_train)
        ind_valid = np.setdiff1d(np.arange(num_examples), ind_train)
        paths_train = np.array(filepaths_all)[ind_train]
        paths_valid = np.array(filepaths_all)[ind_valid]
        labels_train = np.array(labels_all)[ind_train]
        labels_valid = np.array(labels_all)[ind_valid]
        train_generator = DataGenerator(
            paths=paths_train.tolist(),
            labels=labels_train.tolist(),
            batch_size=batch_size,
            preprocess_fn=preprocess_fn)
        valid_generator = DataGenerator(
            paths=paths_valid.tolist(),
            labels=labels_valid.tolist(),
            batch_size=batch_size,
            preprocess_fn=preprocess_fn)
        return train_generator, valid_generator
    else:
        return DataGenerator(
            paths=filepaths_all,
            labels=labels_all,
            batch_size=batch_size,
            preprocess_fn=preprocess_fn)

# import glob
# root_dir = '/Users/fbickfordsmith/Google Drive/Project Code/vgg16'
# glob.iglob(root_dir+'/**', recursive=True) returns an iterator (arbitrary order)
# glob.glob(root_dir+'/**', recursive=True) returns a list (use sorted() to sort)
