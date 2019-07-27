import csv
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

# path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/train/'
# path_contexts = '/home/freddie/attention/contexts/'
# path_synsets = '/home/freddie/attention/metadata/synsets.txt'

path_home = '/Users/fbickfordsmith/Google Drive/Project/'
path_data = path_home+'data/ILSVRC2012_img_val/'
path_contexts = path_home+'attention/contexts/'
path_synsets = path_home+'attention/metadata/synsets.txt'

wnids = [line.rstrip('\n') for line in open(path_synsets)]
wnid2ind = {wnid:ind for ind, wnid in enumerate(wnids)}
generator = ImageDataGenerator().flow_from_directory(directory=path_data)

# data_partition = 'val'
# path_meta = '/home/freddie/attention/metadata/'
# with open(f'{path_meta}{data_partition}_filenames.csv', 'w') as f:
#     wr = csv.writer(f)
#     for item in generator.filenames:
#         wr.writerow([item])

wnids_files = pd.Series(generator.filenames).str.split('/', expand=True)
df = pd.DataFrame()
df['path'] = generator.filenames
df['wnid'] = wnids_files[0]
df['file'] = wnids_files[1]

# for context_type in ['diff', 'sem', 'sim', 'size']:
context_type = 'sem'
path = path_contexts + context_type
with open(f'{path}contexts_wnids.csv') as f:
    contexts = [row for row in csv.reader(f, delimiter=',')]

for i, c in enumerate(contexts):
    inds_c = []
    for wnid in c:
        inds_c.extend(np.flatnonzero(df['wnid']==wnid))
        df.iloc[inds_c].to_csv(
            f'{path}context{i:02}_dataframe.csv', index=False)
