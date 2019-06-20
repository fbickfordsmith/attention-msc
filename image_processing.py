'''
Implement a ImageNet preprocessing routine proposed by Caleb Robinson.

References:
- github.com/calebrob6/imagenet_validation
'''

import cv2
from keras.applications.vgg16 import preprocess_input

# def robinson_processing(img):
#     # img is RGB
#     # load as BGR and convert to RGB: img = cv2.imread(path_img)[:, :, ::-1]
#
#     # resize
#     height = img.shape[0] * 256//min(img.shape[:2])
#     width = img.shape[1] * 256//min(img.shape[:2])
#     img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
#
#     # crop
#     start_row = height//2 - 224//2
#     start_col = width//2 - 224//2
#     end_row = start_row + 224
#     end_col = start_col + 224
#     img = img[start_row:end_row, start_col:end_col]
#
#     # preprocess_input converts to BGR
#     return preprocess_input(img)

def robinson_processing(img):
    # Load as BGR and convert to RGB
#     img = cv2.imread(path_img)[:, :, ::-1]

    # Resize
    height, width, _ = img.shape
    new_height = height * 256//min(img.shape[:2])
    new_width = width * 256//min(img.shape[:2])
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Crop
    height, width, _ = img.shape
    start_x = width//2 - 224//2
    start_y = height//2 - 224//2
    img = img[start_y:(start_y+224), start_x:(start_x+224)]

    # Gets converted to BGR
    return preprocess_input(img)
