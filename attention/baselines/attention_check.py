"""
Assess the accuracy of an attention network with attention weights set to 1.
Written for a sanity check. Agreement with the result produced by
`vgg16_testing.py` implies attention network works as expected.

Method:
1. Load a pretrained VGG16.
2. Add an attention layer after the final pooling layer.
2. Set all attention weights to 1.
3. Fix all weights.
4. Predict the classes of the validation set.
"""

gpu = input('GPU: ')
data_partition = input('Data partition in {train, val, val_white}: ')

import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpu

import numpy as np
from ..utils.paths import path_imagenet
from ..utils.models import build_model
from ..utils.testing import evaluate_model

ind_attention = 19
model = build_model(train=False, attention_position=ind_attention)
predictions, generator = predict_model(
    model, 'dir', path_imagenet/data_partition)
df = evaluate_classwise_accuracy(predictions, generator)
df.to_csv(path_results/'attn_untrained_results.csv', index=False)
mean_acc = np.mean(df['accuracy'])
print(f'Mean accuracy on data partition = {mean_acc}')
