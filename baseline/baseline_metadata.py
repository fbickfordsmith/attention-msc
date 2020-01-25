'''
Use meta.mat from ILSVRC2012_devkit_t12 to add metadata (names and descriptions)
to base_results.csv

References:
- github.com/calebrob6/imagenet_validation/blob/master/1.%20Preprocess%20ImageNet%20validation%20set.ipynb
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import pandas as pd
import scipy.io

path_results = 'results/base_results.csv'
path_meta = 'metadata/meta.mat'
df = pd.read_csv(path_results, index_col=0)
meta = scipy.io.loadmat(path_meta)
ilsvrc2wnid = {}
wnid2name = {}
wnid2desc = {}

for i in range(1000):
    ilsvrc2012_id = int(meta['synsets'][i,0][0][0][0])
    wnid = meta['synsets'][i,0][1][0]
    name = meta['synsets'][i,0][2][0]
    description = meta['synsets'][i,0][3][0]
    ilsvrc2wnid[ilsvrc2012_id] = wnid
    wnid2name[wnid] = name
    wnid2desc[wnid] = description

df['name'] = [wnid2name[wnid] for wnid in df['wnid']]
df['description'] = [wnid2desc[wnid] for wnid in df['wnid']]
df.to_csv(path_results)
