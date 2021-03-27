import os
import pickle

import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, Concatenate
from keras import optimizers
import numpy as np

from util import evaluate_earnings
from data.data_prep import prepare_data
from scipy.signal import butter, lfilter, windows, convolve


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def butter_bandpass(signals, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signals, axis=1)

    return y


def create_dataset(dataset, look_back_1, look_back_2, look_forward):
    window_size = 40
    tau = 5

    dataset_padded = np.pad(dataset, ((int(window_size / 2), int(window_size / 2) - 1), (0, 0)), mode='edge')

    win = windows.exponential(window_size, tau=tau)
    win_past = windows.exponential(window_size, tau=tau)
    win_past[int(window_size / 2):] = 0  # Don't look to future

    signals = np.apply_along_axis(lambda m: convolve(m, win, mode='valid') / sum(win), axis=0, arr=dataset_padded)
    signals_past = np.apply_along_axis(lambda m: convolve(m, win_past, mode='valid') / sum(win_past), axis=0, arr=dataset_padded)
    filtered_sig_grad = np.gradient(signals, axis=0)
    filtered_sig_grad_past = np.gradient(signals_past, axis=0)

    if signals.shape != signals_past.shape != dataset.shape:
        raise Exception('Convolution Error')

    max_look_back = max([look_back_1, look_back_2])

    data_x_1, data_x_2, data_y = [], [], []
    for i in range(filtered_sig_grad.shape[0] - max_look_back - look_forward):
        a = filtered_sig_grad_past[(i + max_look_back - look_back_1):(i + max_look_back), :]
        data_x_1.append(a.T)

        a = filtered_sig_grad_past[(i + max_look_back - look_back_2):(i + max_look_back), :]
        data_x_2.append(a.T)

        data_y.append(sigmoid(np.mean(filtered_sig_grad[(i + max_look_back):(i + max_look_back + look_forward), :], axis=0)))

    data_x_1 = np.asarray(data_x_1)
    data_x_2 = np.asarray(data_x_2)
    data_y = np.asarray(data_y)

    return data_x_1, data_x_2, data_y


def train_lstm(x_train_1, x_train_2, y_train, N_tickers, look_back_1, look_back_2, model_name, normaliser, version):
    # model architecture
    lstm_input_1 = Input(shape=(N_tickers, look_back_1), name='lstm_input_1')
    lstm_input_2 = Input(shape=(N_tickers, look_back_2), name='lstm_input_2')

    x_1 = LSTM(look_back_1, return_sequences=True)(lstm_input_1)
    x_1 = LSTM(look_back_1)(x_1)

    x_2 = LSTM(look_back_2, return_sequences=True)(lstm_input_2)
    x_2 = LSTM(look_back_2)(x_2)

    x = Concatenate(axis=1)([x_1, x_2])

    x = Dropout(0.2)(x)
    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)
    x = Dense(N_tickers)(x)
    output = Activation('sigmoid')(x)

    model = Model(inputs=[lstm_input_1, lstm_input_2], outputs=output)
    adam = optimizers.Adam()
    model.compile(optimizer=adam, loss='mse', metrics=['mae'])
    model.fit(x=[x_train_1, x_train_2], y=y_train, batch_size=32, epochs=400, shuffle=True, validation_split=0.1)

    os.makedirs('models', exist_ok=True)
    model.save(f'models/{model_name}_{version}.h5')
    with open(f'models/{model_name}_normaliser_{version}.pkl', 'wb') as f:
        pickle.dump(normaliser, f)


def load_model(model_name, version):
    if os.path.exists(f'models/{model_name}_{version}.h5'):
        model = keras.models.load_model(f'models/{model_name}_{version}.h5')

    return model


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
    version = 'dev'

    train_dates, test_dates, train, test, train_normalised, test_normalised, N_tickers, normaliser, symbols = prepare_data(sector, load_normaliser=False, version=version)

    look_back_1 = 40
    look_back_2 = 80
    look_forward = 2
    x_train_1, x_train_2, y_train = create_dataset(train_normalised, look_back_1, look_back_2, look_forward)
    x_test_1, x_test_2, y_test = create_dataset(test_normalised, look_back_1, look_back_2, look_forward)

    max_look_back = max([look_back_1, look_back_2])

    dates = test_dates[max_look_back:-look_forward]
    y_test_real = test[max_look_back:-look_forward]

    if mode == 'train':
        if model_type == 'lstm':
            train_lstm(x_train_1, x_train_2, y_train, N_tickers, look_back_1, look_back_2, model_name, normaliser, version)

    model = load_model(model_name, version)

    y_test_preds = model.predict([x_test_1, x_test_2])

    earnings, index_earnings = [], []
    for target_stock in range(N_tickers):
        print(f"Target Stock: {symbols[target_stock]}")
        earning, index_earning = evaluate_earnings(dates, y_test_preds, dates, y_test, dates, y_test_real, target_stock, symbols[target_stock])
        earnings.append(earning)
        index_earnings.append(index_earning)

    import matplotlib.pyplot as plt
    plt.hist(earnings, bins=30, label='Algorithm')
    plt.hist(index_earnings, bins=30, label='Index')
    plt.legend()
    plt.show()

    print(f"Average Earnings: {np.mean(earnings)}")
    print(f"Average Index Performance: {np.mean(index_earnings)}")
