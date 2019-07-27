'''
Implement ImageNet preprocessing routine used by
- Caleb Robinson (github.com/calebrob6/imagenet_validation)
- Ken Luo (github.com/don-tpanic/CSML_attention_project_pieces)
'''

import cv2
from keras.applications.vgg16 import preprocess_input
import numpy as np

def robinson_processing(img):
    # resize
    height = img.shape[0] * 256//min(img.shape[:2])
    width = img.shape[1] * 256//min(img.shape[:2])
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

    # crop
    start_row = height//2 - 224//2
    start_col = width//2 - 224//2
    end_row = start_row + 224
    end_col = start_col + 224
    img = img[start_row:end_row, start_col:end_col]

    # img is RGB; preprocess_input converts to BGR
    return preprocess_input(img)

def random_crop_batch(batch, random_crop_size):
    height, width = batch.shape[1], batch.shape[2]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return batch[:, y:(y+dy), x:(x+dx), :]

def pca_augment(inputs, std_deviation=0.1, scale=1.0, clipping=False):
    ranks = inputs.ndim
    assert ranks >= 2

    chs = inputs.shape[-1]

    # swapaxis, reshape for calculating covariance matrix
    # rank 2 = (batch, dims)
    # rank 3 = (batch, step, dims)
    if ranks <= 3:
        x = inputs.copy()
    # rank 4 = (batch, height, width, ch) -> (batch, dims, ch)
    elif ranks == 4:
        dims = inputs.shape[1] * inputs.shape[2]
        x = inputs.reshape(-1, dims, chs)
    # rank 5 = (batch, D, H, W, ch) -> (batch, D, dims, ch)
    elif ranks == 5:
        dims = inputs.shape[2] * inputs.shape[3]
        depth = inputs.shape[1]
        x = inputs.reshape(-1, depth, dims, chs)

    # scaling-factor
    calculate_axis, reduce_axis = ranks-1, ranks-2
    if ranks == 3:
        calculate_axis, reduce_axis = 1, 2
    elif ranks >= 4:
        calculate_axis, reduce_axis = ranks-3, ranks-2
    C = 1.0
    if ranks >= 3:
        C = x.shape[reduce_axis]

    ###########################################################################
    ### normalize x by using mean and std
    # variance within each chl
    var = np.var(x, axis=calculate_axis, keepdims=True)
    # 1./std along each chl
    scaling_factors = np.sqrt(C / np.sum(var, axis=reduce_axis, keepdims=True))
    # scaling
    x = x * scaling_factors
    # subtract mean for cov matrix
    mean = np.mean(x, axis=calculate_axis, keepdims=True)
    x -= mean
    ###########################################################################
    # covariance matrix
    cov_n = max(x.shape[calculate_axis] - 1, 1)
    # cov (since x was normalized --> x.T * x gives the var-cov matrix)
    cov = np.matmul(np.swapaxes(x, -1, -2), x) / cov_n

    # eigen value(S), eigen vector(U)
    U, S, V = np.linalg.svd(cov)

    # random values
    # if rank2 : get differnt random variable by sample
    if ranks == 2:
        rand = np.random.randn(*inputs.shape) * std_deviation
        delta = np.matmul(rand*np.expand_dims(S, axis=0), U)
    else:
        # rand -> size=len(S), random int between low and high eigenvalues, multiply std
        rand = np.random.randn(*S.shape) * std_deviation
        # [p1, p2, p3][a1r1, a2r2, a3r3].T
        delta_original = np.squeeze(np.matmul(U, np.expand_dims(rand*S, axis=-1)), axis=-1)

    # adjust delta shape
    if ranks == 3:
        delta = np.expand_dims(delta_original, axis=ranks-2)
    elif ranks >= 4:
        delta = np.expand_dims(delta_original, axis=ranks-3)
        delta = np.broadcast_to(delta, x.shape)
        delta = delta.reshape(-1, *inputs.shape[1:])

    # delta scaling
    delta = delta * scale

    result = inputs + delta
    if clipping:
        """
        vgg16 does not clip:
        https://arxiv.org/pdf/1409.1556.pdf
        """
        result = np.clip(result, 0.0, scale)

    return result

def crop_and_pca_generator(generator, crop_length):
    while True:
        batch_x, batch_y = next(generator)
        batch_crops = np.zeros((batch_x.shape[0], crop_length, crop_length, 3))

        # crop by bacth:
        batch_size = batch_x.shape[0]
        batch_crops = random_crop_batch(batch_x, (crop_length, crop_length))

        # pca-aug by batch
        batch_crops = pca_augment(batch_crops)
        yield (batch_crops, batch_y)
