import pickle

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from model.inception_v3_fc import *
from data import load_data, load_train_generator


def train_inceptionv3_baseline(train_data_file, val_data_file, ckpt_file, hidden_size):
    print('Building model ...')
    model = inception_v3_single_fc_model(hidden_size)
    _train(model, train_data_file, val_data_file, ckpt_file)


def train_inceptionv3_mlp(train_data_file, val_data_file, ckpt_file, hidden_sizes, drop_rate=0.2):
    print('Building model ...')
    model = inception_v3_muliple_fc_drop_model(hidden_sizes, drop_rate)
    _train(model, train_data_file, val_data_file, ckpt_file)


def _train(model, train_data_file, val_data_file, ckpt_file):
    train_datagen = load_train_generator(train_data_file, batch_size=50)
    x_val, y_val = load_data(val_data_file)
    ckbt_callback = ModelCheckpoint(ckpt_file)
    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy')
    model.fit_generator(train_datagen, samples_per_epoch=5000, nb_epoch=5,
                        validation_data=(x_val, y_val), callbacks=[ckbt_callback])
