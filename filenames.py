from keras.preprocessing.image import ImageDataGenerator

path_data = '/mnt/fast-data16/datasets/ILSVRC/2012/clsloc/'

datagen = ImageDataGenerator()

for data_partition in ('train', 'val', 'val_white'):
    generator = datagen.flow_from_directory(
        directory=path_data+data_partition,
        target_size=(224, 224),
        batch_size=1,
        shuffle=False,
        class_mode=None)

    filenames = np.array(generator.filenames)

    np.save(
        f'{path_activations}class{i:04}_activations.npy'
        class_activations,
        allow_pickle=False)
