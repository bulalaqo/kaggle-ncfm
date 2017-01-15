import pickle

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.cross_validation import KFold

from model.inception_v3_feat import *
from data import load_data, load_train_generator


def extract_feature(train_data_file, feature_file):
    x_train, y_train = load_data(train_data_file)
    model = inception_v3_avg_feat_model()
    features = model.predict(x_train, batch_size=100)
    with open(feature_file, 'wb') as f:
        pickle.dump(obj=(features, y_train), file=f)


def train_linear(feature_file):
    with open(feature_file, 'rb') as f:
        x_all, y_all = pickle.load(f)

    kf = KFold(len(y_all), n_folds=5, random_state=791211)
    for train_index, val_index in kf:
        model = linear_model(input_size=x_all.shape[1])
        model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        x_train, x_val = x_all[train_index], x_all[val_index]
        y_train, y_val = y_all[train_index], y_all[val_index]
        model.fit(x_train, y_train, batch_size=50, nb_epoch=5, validation_data=(x_val, y_val))
