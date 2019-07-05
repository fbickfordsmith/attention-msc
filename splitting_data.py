'''
Take a CSV where each row is a list of WordNet IDs (ImageNet class names) like
['n01440764', 'n01438464', ...].

For each set, i,
- Move the folders for the in-set classes into a new folder called `seti`.
- In the `seti` folder, create empty folders for all out-of-set classes.

Command-line arguments:
1. data_partition in {train, val, val_white}
2. type_context in {diff, sim}

References:
- stackoverflow.com/questions/15034151/copy-directory-contents-into-a-directory-with-python
- thispointer.com/how-to-create-a-directory-in-python/
- thispointer.com/python-how-to-move-files-and-directories/
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import sys
import shutil
from glob import glob
from distutils.dir_util import copy_tree
import numpy as np

script_name, data_partition, type_context = sys.argv
path_home = '/home/freddie/'
path_data = f'{path_home}ILSVRC2012-{type_context}contexts/{data_partition}/'
path_wnids = f'{path_home}attention/csv/{type_context}contexts_wnids.csv'
path_allclasses = f'{path_home}attention/txt/synsets.txt'
contexts = np.loadtxt(path_wnids, dtype=str, delimiter=',')
all_classes = np.array([line.rstrip('\n') for line in open(path_allclasses)])
print(f'Running {script_name} on {path_data}')

for i, incontext_classes in enumerate(contexts):
    context_folder = f'context{i:02}/'
    for incontext_class in incontext_classes:
        # copy incontext_class folder to the folder for this context
        copy_tree(
            path_data+incontext_class, # source path
            path_data+context_folder+incontext_class) # destination path

    # np.setdiff1d(a, b) returns unique values in a that are not in b
    outofcontext_classes = np.setdiff1d(all_classes, incontext_classes)
    for outofcontext_class in outofcontext_classes:
        # make new empty folders for all classes not in this context
        os.makedirs(path_data+context_folder+outofcontext_class)

# remove the original folders
folders_keep = [f'{path_data}context{i:02}/' for i in range(contexts.shape[0])]
folders_remove = [g for g in glob(f'{path_data}*/') if g not in folders_keep]
for f in folders_remove:
    shutil.rmtree(f)
