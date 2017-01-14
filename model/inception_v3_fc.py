import pdb
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import AveragePooling2D
from keras.models import Model

from model.inception_v3 import InceptionV3
from config import NUM_CLASS


def inception_v3_single_fc_model(hidden_size):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    net = Flatten()(base_model.output)
    net = Dense(NUM_CLASS, activation='softmax')(net)
    return Model(base_model.input, net)


def inception_v3_muliple_fc_model(hidden_sizes):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    net = AveragePooling2D((8, 8), strides=(8, 8))(base_model.output)
    net = Flatten()(net)
    for hidden_size in hidden_sizes:
        net = Dense(hidden_size, activation='relu')(net)
    net = Dense(NUM_CLASS, activation='softmax')(net)
    return Model(base_model.input, net)


def inception_v3_muliple_fc_drop_model(hidden_sizes, drop_rate=0.2):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    net = AveragePooling2D((8, 8), strides=(8, 8))(base_model.output)
    net = Flatten()(net)
    for hidden_size in hidden_sizes:
        net = Dense(hidden_size, activation='relu')(net)
    net = Dropout(drop_rate)(net)
    net = Dense(NUM_CLASS, activation='softmax')(net)
    return Model(base_model.input, net)
