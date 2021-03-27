import pickle

import pandas as pd
from sklearn import preprocessing


def csvs_to_dataset(sector, start_date, end_date, save_model_stocks, load_normaliser, version):
    master_df = pd.read_csv(f"data/{sector}_master.csv")
    master_df['date'] = pd.to_datetime(master_df['date'], format='%Y-%m-%d')

    master_df = master_df[master_df['date'] > pd.to_datetime(start_date, format='%Y-%m-%d')]
    master_df = master_df[master_df['date'] < pd.to_datetime(end_date, format='%Y-%m-%d')]

    cols_to_drop = [col for col in list(master_df) if pd.isna(master_df.iloc[0][col])]
    master_df.drop(columns=cols_to_drop, inplace=True)

    if save_model_stocks:
        symbols = [symbol for symbol in list(master_df) if symbol != 'date']
        with open(f'models/{sector}_symbols_{version}.pkl', 'wb') as f:
            pickle.dump(symbols, f)
    else:
        with open(f'models/{sector}_symbols_{version}.pkl', 'rb') as f:
            symbols = pickle.load(f)

    master_df = master_df[['date'] + symbols]

    master_df = master_df.interpolate(method='linear', axis=0).ffill().bfill()

    master_df.set_index('date', inplace=True)

    dates = master_df.index.to_numpy()

    data = master_df.to_numpy()

    symbols = list(master_df)

    if not load_normaliser:
        data_normaliser = preprocessing.StandardScaler()
        data_normalised = data_normaliser.fit_transform(data)
    else:
        with open(f'models/{sector}_lstm_normaliser_{version}.pkl', 'rb') as f:
            data_normaliser = pickle.load(f)
        data_normalised = data_normaliser.transform(data)

    return dates, data, data_normalised, data_normaliser, symbols


def prepare_data(sector, split=True, start_date='2012-07-01', end_date='2019-01-01', save_model_stocks=True, load_normaliser=False, version='prod'):
    dates, close_prices, normalised_close_prices, normaliser, symbols = csvs_to_dataset(sector, start_date, end_date, save_model_stocks, load_normaliser, version)

    N_tickers = close_prices.shape[1]

    if split:
        # split into train and test sets
        train_frac = 0.8
        train_size = int(len(normalised_close_prices) * train_frac)
        train_dates, test_dates = dates[:train_size], dates[train_size:]
        train, test = close_prices[:train_size, :], close_prices[train_size:, :]
        train_normalised, test_normalised = normalised_close_prices[:train_size, :], normalised_close_prices[train_size:, :]

        return train_dates, test_dates, train, test, train_normalised, test_normalised, N_tickers, normaliser, symbols

    else:
        return dates, close_prices, normalised_close_prices, N_tickers, normaliser, symbols
