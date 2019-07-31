import sys
import os
import csv
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

_, data_partition = sys.argv

path_data = f'/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/{data_partition}/'
path_contexts = '/home/freddie/attention/contexts/'
path_synsets = '/home/freddie/attention/metadata/synsets.txt'
path_save = f'/home/freddie/dataframes_{data_partition}/'

# path_data = '/Users/fbickfordsmith/Google Drive/Project/data/ILSVRC2012_img_val/'
# path_contexts = '/Users/fbickfordsmith/Google Drive/Project/attention/contexts/'
# path_synsets = '/Users/fbickfordsmith/Google Drive/Project/attention/metadata/synsets.txt'

wnids = [line.rstrip('\n') for line in open(path_synsets)]
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}
generator = ImageDataGenerator().flow_from_directory(directory=path_data)

wnids_files = pd.Series(generator.filenames).str.split('/', expand=True)
df = pd.DataFrame()
df['path'] = generator.filenames
df['wnid'] = wnids_files[0]
df['file'] = wnids_files[1]

for type_context in ['diff', 'sem', 'sim', 'size']:
    print(type_context)

    with open(f'{path_contexts}{type_context}contexts_wnids.csv') as f:
        contexts = [row for row in csv.reader(f, delimiter=',')]

    os.makedirs(f'{path_save}{type_context}contexts')

    for i, c in enumerate(contexts):
        inds_c = []
        for wnid in c:
            inds_c.extend(np.flatnonzero(df['wnid']==wnid))
            df.iloc[inds_c].to_csv(
                f'{path_save}{type_context}contexts/'
                f'{type_context}context{i:02}_dataframe.csv',
                index=False)

# data_partition = 'val'
# path_meta = '/home/freddie/attention/metadata/'
# with open(f'{path_meta}{data_partition}_filenames.csv', 'w') as f:
#     wr = csv.writer(f)
#     for item in generator.filenames:
#         wr.writerow([item])
