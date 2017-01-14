import pickle

from keras.optimizers import Adam

from model.inception_v3_fc import inception_v3_single_fc_model


def train_inceptionv3_baseline(train_data_file, val_data_file, hidden_size):
    print('Loading data ...')
    with open(train_data_file, 'rb') as f:
        x_train, y_train = pickle.load(f)
    print('Train data: {}'.format(x_train.shape))

    with open(val_data_file, 'rb') as f:
        x_val, y_val= pickle.load(f)
    print('Val data: {}'.format(x_val.shape))

    print('Building model ...')
    model = inception_v3_single_fc_model(hidden_size)

    model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy')
    model.fit(x_train, y_train, batch_size=50, nb_epoch=5, validation_data=(x_val, y_val))
    