'''
Assess the accuracy of an attention network with attention weights set to 1.
Written for a sanity check. Agreement with the result produced by
`baseline_average.py` implies attention network works as expected.

Method:
1. Load a pretrained VGG16.
2. Add an attention layer after the final pooling layer.
2. Set all attention weights to 1.
3. Fix all weights.
4. Predict the classes of the validation set.
'''

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = input('GPU: ')

import numpy as np
from layers import Attention
from models import build_model
from testing import evaluate_model

data_partition = 'val_white'
path_data = f'/fast-data/datasets/ILSVRC/2012/clsloc/{data_partition}/'
model = build_model(train=False)
model.layers[19].set_weights([np.ones((1, 7, 7, 512))])
scores = evaluate_model(model, path_data)
print(f'{model.metrics_names} = {scores}')
