import pdb
from keras.layers import Dense, Flatten
from keras.models import Model

from model.inception_v3 import InceptionV3
from config import NUM_CLASS


def inception_v3_single_fc_model(hidden_size):
    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    net = Flatten()(base_model.output)
    net = Dense(NUM_CLASS, activation='softmax')(net)
    return Model(base_model.input, net)