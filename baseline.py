import pickle

from keras.optimizers import Adam

from model.inception_v3_fc import inception_v3_single_fc_model
from data import load_train_val_data

def train_inceptionv3_baseline(train_data_file, val_data_file, hidden_size):
    x_train, y_train, x_val, y_val = load_train_val_data(train_data_file, val_data_file)

    print('Building model ...')
    model = inception_v3_single_fc_model(hidden_size)

    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, batch_size=50, nb_epoch=5, validation_data=(x_val, y_val))
