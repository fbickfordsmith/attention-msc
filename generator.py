'''
Define a data generator based on keras.utils.Sequence.

Partitioning data:
    import pandas as pd
    from keras.preprocessing.image import ImageDataGenerator

    data_partition = 'train'
    path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
# path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
    path_activations = '/home/freddie/activations-conv-split/'
    path_synsets = '/home/freddie/attention/metadata/synsets.txt'

    generator = ImageDataGenerator().flow_from_directory(directory=path_data)
    filepaths = path_activations + pd.Series(generator.filenames).str.replace('.JPEG', '_conv5.npy')
    path2label = {filepath:label for filepath, label in zip(filepaths, generator.classes)}
    wnids = open(path_synsets).read().splitlines()
    wnid2ind = generator.class_indices
    split = 0.1

    partition = {'train':[], 'valid':[]}
    for wnid in wnids:
        inds = np.flatnonzero(generator.classes==wnid2ind[wnid])
        val_size = int(split*len(inds))
        partition['train'].extend(filepaths[inds][val_size:])
        partition['valid'].extend(filepaths[inds][:val_size])

    generator_train = DataGenerator(partition['train'], path2label)
    generator_valid = DataGenerator(partition['valid'], path2label)

References:
- stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
- keras.io/utils/
'''

import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, paths, path2label, batch_size=256, num_classes=1000, shuffle=True):
        self.paths = paths
        self.path2label = path2label
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.n = len(self.paths)
        self.on_epoch_end()

    def __len__(self):
        # return number of batches per epoch
        return int(np.ceil(len(self.paths)/self.batch_size))

    def __getitem__(self, index):
        # return a batch of data
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        paths_temp = [self.paths[i] for i in indices]
        return self.__data_generation(paths_temp)

    def on_epoch_end(self):
        # update indices
        self.indices = np.arange(len(self.paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, paths_temp):
        X = np.array([np.load(path) for path in paths_temp])
        y = np.array([self.path2label[path] for path in paths_temp])
        return X, y
