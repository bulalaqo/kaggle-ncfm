import pdb
from keras.layers import Dense, Flatten, Dropout, Input
from keras.layers.convolutional import AveragePooling2D
from keras.models import Model

from config import NUM_CLASS
from model.inception_v3 import InceptionV3


def inception_v3_avg_feat_model():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    net = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(base_model.output)
    net = Flatten(name='flatten')(net)
    return Model(base_model.input, net)


def inception_v3_feat_model():
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    net = Flatten(name='flatten')(base_model.output)
    return Model(base_model.input, net)


def linear_model(input_size):
    input = Input(shape=(input_size,))
    net = Dense(NUM_CLASS, activation='softmax')(input)
    return Model(input, net)
