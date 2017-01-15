import pickle

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from model.inception_v3_feat import *
from data import load_data, load_train_generator


def extract_feature(train_data_file, feature_file):
    x_train, y_train = load_data(train_data_file)
    model = inception_v3_avg_feat_model()
    features = model.predict(x_train, batch_size=100)
    with open(feature_file, 'wb') as f:
        pickle.dump(obj=(features, y_train), file=f)
