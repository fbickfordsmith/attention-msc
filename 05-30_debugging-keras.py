import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

'''
Use a pretrained VGG16 to classify ImageNet examples.

Reference: https://medium.com/@vijayabhaskar96/tutorial-image-classification-
    with-keras-flow-from-directory-and-generators-95f75ebe5720
'''

import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array





# use decode_predictions to get predicted classes
# we know true classes from folders


import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

model = VGG16(weights='imagenet')

base_path = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/val_white'
folders = sorted(os.listdir(base_path))
img_paths = [
    base_path+'/'+folder+'/'+img_name
    for folder in folders if not folder.startswith('.')
    for img_name in sorted(os.listdir(base_path+'/'+folder))]

ind2name = {ind:name for ind, name in enumerate(folders)}

class_sizes = [
    len(os.listdir(base_path+'/'+folder))
    for folder in os.listdir(base_path) if not folder.startswith('.')]

predicted_classes = []
for i, img_path in enumerate(img_paths):
    if i % 2000 == 0:
        print(f'{i:05} images processed')
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    predicted_classes.append(np.argmax(model.predict(img)))

datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
generator = datagen.flow_from_directory(
    directory=path,
    target_size=(224, 224),
    batch_size=1,
    shuffle=False, # False => returns images in order
    class_mode=None) # None => returns just images (no labels)
true_classes = generator.classes
name2ind = generator.class_indices
ind2name = {ind:name for name, ind in name2ind.items()}
predicted_names = np.array([ind2name[ind] for ind in predicted_classes])

true_names_rep = np.array([])
for c in true_classes:
    repeated =
    true_names_rep = np.concatenate((true_names_rep, ))



num_examples = len(true_classes)
predicted_probs = model.predict_generator(
    generator,
    steps=num_examples//batch_size,
    verbose=True)
predicted_classes = np.argmax(predicted_probs, axis=1)
correct = (predicted_classes==true_classes)
accuracy = np.mean(correct)
print(f'Top-1 accuracy: {(accuracy*100):.2f}%')

ind_correct = np.flatnonzero(correct)
classes_correct = true_classes[ind_correct]
# Alternative/equivalent method
# classes_correct = (correct * (true_classes+1)) - 1 # -1 if incorrect; else class

counts_correct = np.array(
    [np.count_nonzero(classes_correct==i) for i in range(1000)])
counts_classes = np.array(
    [np.count_nonzero(true_classes==i) for i in range(1000)])
accuracy_classes = counts_correct / counts_classes
np.savetxt('accuracy_classes.csv', accuracy_classes)

# Code for converting indices to class IDs
# name2ind = generator.class_indices
# ind2name = {ind:name for name, ind in name2ind.items()}
# predicted_names = [ind2name[ind] for ind in predicted_labels]
