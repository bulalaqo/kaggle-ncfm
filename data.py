import glob
import numpy as np
import os
import pickle
from scipy.misc import imread, imresize
from sklearn.utils import shuffle

from keras.preprocessing.image import ImageDataGenerator

from config import CLASS_MAP


def data_stats(train_path, test_path):
    train_sub_paths = glob.glob(os.path.join(train_path, '*'))
    for train_sub_path in train_sub_paths:
        cls_name = train_sub_path.split()[-1]
        train_img_files = glob.glob(os.path.join(train_sub_path, '*'))
        print('{}: {}'.format(cls_name, len(train_img_files)))

    test_img_files = glob.glob(os.path.join(test_path, '*'))
    print('Test: {}'.format(len(test_img_files)))


def compile_train_val_file(data_path, dump_train_file, dump_val_file, size=(299, 299)):
    imgs = []
    labels = []
    train_sub_paths = sorted(glob.glob(os.path.join(data_path, '*')))
    for train_sub_path in train_sub_paths:
        print('Processing {} ...'.format(train_sub_path))
        cls_name = train_sub_path.split('/')[-1]
        train_img_files = sorted(glob.glob(os.path.join(train_sub_path, '*')))
        for train_img_file in train_img_files:
            print(train_img_file, end='\r')
            origin_im = imread(train_img_file)
            resize_im = imresize(origin_im, size)
            imgs.append(resize_im)
            labels.append(CLASS_MAP[cls_name])

    train_size = len(imgs) * 4 // 5
    imgs, labels = shuffle(imgs, labels)
    train_imgs = np.stack(imgs[:train_size])
    train_labels = np.stack(labels[:train_size])
    val_imgs = np.stack(imgs[train_size:])
    val_labels = np.stack(labels[train_size:])

    print('Dump train data {} to {}'.format(train_imgs.shape, dump_train_file))
    with open(dump_train_file, 'wb') as f:
        pickle.dump(obj=(train_imgs, train_labels), file=f)

    print('Dump val data {} to {}'.format(val_imgs.shape, dump_val_file))
    with open(dump_val_file, 'wb') as f:
        pickle.dump(obj=(val_imgs, val_labels), file=f)


def load_train_generator(train_data_file, batch_size=50):
    """Random flipping, corping"""
    x_train, y_train = load_data(train_data_file)
    gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                             horizontal_flip=True, vertical_flip=True)
    return gen.flow(x_train, y_train, batch_size=batch_size)


def load_data(data_file):
    print('Loading data from {} ...'.format(data_file))
    with open(data_file, 'rb') as f:
        features, labels = pickle.load(f)
        features = inception_preprocess(features)
    print('Data: {}'.format(features.shape))
    return features, labels


def inception_preprocess(x):
    x = x.astype(np.float32)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x