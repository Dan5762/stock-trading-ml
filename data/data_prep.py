import os

import pandas as pd
import numpy as np
from sklearn import preprocessing


def csvs_to_dataset(csv_folder):
    data_agg = []
    for filename in os.listdir(csv_folder):
        csv_path = os.path.join(csv_folder, filename)

        data = pd.read_csv(csv_path, header=None)

        data = data.to_numpy().reshape((-1,))

        if len(data) == 2919:
            data_agg.append(data)
        else:
            print(filename)
            print(len(data))
            print()

    data = np.stack(data_agg, axis=1)

    data_normaliser = preprocessing.StandardScaler()
    data_normalised = data_normaliser.fit_transform(data)

    return data, data_normalised, data_normaliser


def prepare_data(sector):
    # dataset
    close_prices, normalised_close_prices, normaliser = csvs_to_dataset(f'data/{sector}')

    N_tickers = close_prices.shape[1]

    # split into train and test sets
    train_frac = 0.8
    train_size = int(len(normalised_close_prices) * train_frac)
    train, test = close_prices[:train_size, :], close_prices[train_size:, :]
    train_normalised, test_normalised = normalised_close_prices[:train_size, :], normalised_close_prices[train_size:, :]

    return train, test, train_normalised, test_normalised, N_tickers, normaliser
