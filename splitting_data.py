'''
Take a CSV where each row is a list of WordNet IDs (ImageNet class names) like
['n01440764', 'n01438464', ...].

For each context, i,
- Copy the folders for the in-context classes into a new folder called `contexti`.
- In the `contexti` folder, create empty folders for all out-of-context classes.

Command-line arguments:
1. data_partition in {train, val, val_white}
2. type_context in {diff, sim, sem, size}

References:
- stackoverflow.com/questions/15034151/copy-directory-contents-into-a-directory-with-python
- thispointer.com/how-to-create-a-directory-in-python/
- thispointer.com/python-how-to-move-files-and-directories/
'''

import sys
import csv
import os
import shutil
from glob import glob
from distutils.dir_util import copy_tree
import numpy as np

script_name, data_partition, type_context = sys.argv
path_data = f'/home/freddie/ILSVRC2012-{type_context}contexts/{data_partition}/'
path_contexts = f'/home/freddie/attention/contexts/{type_context}contexts_wnids.csv'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'
with open(path_contexts) as f:
    contexts = [row for row in csv.reader(f, delimiter=',')]
all_classes = np.array([line.rstrip('\n') for line in open(path_synsets)])
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
folders_keep = [f'{path_data}context{i:02}/' for i in range(len(contexts))]
folders_remove = [g for g in glob(f'{path_data}*/') if g not in folders_keep]
for f in folders_remove:
    shutil.rmtree(f)
