from collections import defaultdict
import glob
import numpy as np
import os
import pdb
import pickle
import random
from scipy.misc import imread, imresize
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from sklearn.decomposition import RandomizedPCA
from keras.preprocessing.image import ImageDataGenerator

from config import CLASS_MAP


def cluster_train_boat(data_path, dump_train_file, size=(299, 299)):
    imgs = []
    imgs_for_clustering = []
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
            normalize_im = imresize(origin_im, (32, 32)).astype(np.float32) / 255.
            imgs_for_clustering.append(normalize_im)
            imgs.append(resize_im)
            labels.append(CLASS_MAP[cls_name])

    stack_imgs = np.stack(imgs_for_clustering)
    flatten_imgs = stack_imgs.reshape((len(imgs_for_clustering), -1))

    print('PCA ...')
    model = RandomizedPCA(50, whiten=True)
    pca_imgs = model.fit_transform(flatten_imgs)

    print('Kmeans ...')
    kmeans = KMeans(n_clusters=12, random_state=0).fit(pca_imgs)
    predict_boat = kmeans.predict(pca_imgs)

    boat_data = defaultdict(list)
    for boat, img, label in zip(predict_boat, imgs, labels):
        boat_data[boat].append((img, label))

    boat_train_data = list(boat_data.values())
    with open(dump_train_file, 'wb') as f:
        pickle.dump(obj=boat_train_data, file=f)


class BoatTrainData:
    def __init__(self, train_file):
        with open(train_file, 'rb') as f:
            self.data = pickle.load(f)
        print([len(x) for x in self.data])


    def get_train_val_set(self, seed=0, batch_size=50):
        random.seed(seed)
        idxs = [i for i in range(len(self.data)) if len(self.data[i]) < 400]
        val_size = 0
        val_idx = []
        while val_size < 500:
            val_idx.append(random.choice(idxs))
            val_size += self.data[val_idx]

        x_train, y_train, x_val, y_val = [], [], [], []
        for i, sub_data in enumerate(self.data):
            if i in val_idx:
                for x, y in sub_data:
                    x_val.append(x)
                    y_val.append(y)
            else:
                for x, y in sub_data:
                    x_train.append(x)
                    y_train.append(y)
        x_train = inception_preprocess(np.stack(x_train))
        x_val = inception_preprocess(np.stack(x_val))
        train_gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                       horizontal_flip=True, vertical_flip=True)

        return train_gen.flow(x_train, y_train, batch_size=batch_size), x_val, y_val


def inception_preprocess(x):
    x = x.astype(np.float32, copy=False)
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
