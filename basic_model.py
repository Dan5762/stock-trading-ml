import os
import pickle

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
from scipy import signal

from utils.pyESN import ESN 
from util import evaluate_earnings_classifier
from data.data_prep import prepare_data


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def create_dataset(dataset, look_back, look_forward):
    window_size = 50

    print(dataset.shape)

    dataset_padded = np.pad(dataset, ((int(window_size // 2), int(window_size // 2) - 1), (0, 0)), mode='edge')

    win = signal.windows.hann(window_size)
    signals = np.apply_along_axis(lambda m: signal.convolve(m, win, mode='valid') / sum(win), axis=0, arr=dataset_padded)
    filtered_sig_grad = np.gradient(signals, axis=0)

    if signals.shape != dataset.shape:
        raise Exception('Convolution Error')

    data_x, data_y = [], []
    for i in range(filtered_sig_grad.shape[0] - look_back - look_forward - 1):
        a = filtered_sig_grad[i:(i + look_back), :]
        data_x.append(a.T)
        data_y.append(sigmoid(np.mean(filtered_sig_grad[(i + look_back):(i + look_back + look_forward), :], axis=0)))

    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)

    return data_x, data_y


def train_lstm(x_train, y_train, N_tickers, look_back, model_name):
    # model architecture
    lstm_input = Input(shape=(N_tickers, look_back), name='lstm_input')
    x = LSTM(look_back, return_sequences=True)(lstm_input)
    x = LSTM(look_back)(x)
    x = Dropout(0.2)(x)
    x = Dense(32)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)
    x = Dense(N_tickers)(x)
    output = Activation('sigmoid')(x)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mae', metrics=['mse'])
    history = model.fit(x=x_train, y=y_train, batch_size=32, epochs=200, shuffle=True, validation_split=0.1)

    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model_name}.h5')
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(history.history, f)


def train_echo_net(x_train, y_train, model_name):
    # model architecture
    model = ESN(n_inputs=1,
                n_outputs=1,
                n_reservoir=500,
                sparsity=0.2,
                random_state=23,
                spectral_radius=1.2,
                noise=.0005)

    model.fit(x_train, y_train)

    os.makedirs('models', exist_ok=True)
    with open(f'models/{model_name}.pkl', 'wb') as f:
        pickle.dump(model, f)


def load_model(model_name):
    if os.path.exists(f'models/{model_name}.h5'):
        model = keras.models.load_model(f'models/{model_name}.h5')
        with open(f'models/{model_name}.pkl', 'rb') as f:
            history = pickle.load(f)

    elif os.path.exists(f'models/{model_name}.pkl'):
        with open(f'models/{model_name}.pkl', 'rb') as f:
            model = pickle.load(f)
        history = []

    return model, history


def pad_and_unscale(scaled_predictions, normaliser, target_stock):
    scaled_predictions_padded = np.zeros((scaled_predictions.shape[0], N_tickers))
    scaled_predictions_padded[:, target_stock] = scaled_predictions.reshape((-1,))
    unscaled_predictions = normaliser.inverse_transform(scaled_predictions_padded)
    unscaled_predictions = unscaled_predictions[:, target_stock]

    return unscaled_predictions


if __name__ == "__main__":
    mode = 'load'
    sector = 'tech'
    model_type = 'lstm'
    model_name = 'tech_lstm'

    train, test, train_normalised, test_normalised, N_tickers, normaliser = prepare_data(sector)

    look_back = 50
    look_forward = 10
    x_train, y_train = create_dataset(train_normalised, look_back, look_forward)
    x_test, y_test = create_dataset(test_normalised, look_back, look_forward)

    train_real_price = train[look_back:]
    test_real_price = test[look_back:]

    if mode == 'train':
        if model_type == 'lstm':
            train_lstm(x_train, y_train, N_tickers, look_back, model_name)
        elif model_type == 'echo_net':
            train_echo_net(x_train, y_train, model_name)

    model, history = load_model(model_name)

    # evaluation
    y_test_predicted = model.predict(x_test)
    y_train_predicted = model.predict(x_train)

    x_test_unscaled = []
    for test_set in x_test:
        x_test_unscaled.append(normaliser.inverse_transform(test_set.T))
    x_test_unscaled = np.asarray(x_test_unscaled)

    earnings, index_earnings = [], []
    for target_stock in range(N_tickers):
        print(f"Target Stock: {target_stock}")
        earning, index_earning = evaluate_earnings_classifier(x_test, y_test, model, target_stock, test_real_price[:-look_forward, :])
        earnings.append(earning)
        index_earnings.append(index_earning)

    print(f"Average Earnings: {np.mean(earnings)}")
    print(f"Average Index Performance: {np.mean(index_earning)}")
