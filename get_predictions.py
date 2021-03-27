import pickle
import pprint as pp
from datetime import timedelta, date
from datetime import datetime

import numpy as np
import pandas as pd

from data.data_prep import prepare_data
from data.load_data import get_sector
from basic_model import create_dataset, load_model
from util import evaluate_earnings

look_back_1 = 40
look_back_2 = 80
look_forward = 5
model_type = 'lstm'
model_name = 'tech_lstm'
sector = 'tech'
version = 'dev'
update = False

purchase_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'NVDA', 'PYPL', 'INTC', 'CRM', 'AMD', 'ATVI', 'MTCH',
                    'EA', 'ZG', 'TTD', 'YELP', 'ADBE', 'CSCO', 'ASML', 'ORCL', 'AVGO', 'TXN', 'SHOP', 'QCOM', 'SAP', 'SQ', 
                    'INTU', 'ZM', 'AMAT', 'NOW', 'IBM']

chosen_stocks = ['QCOM', 'TSLA', 'AVGO', 'ASML', 'TXN', 'ADBE', 'MSFT', 'IBM', 'AAPL', 'AMAT']

pp = pp.PrettyPrinter(indent=4)

if __name__ == "__main__":
    with open(f'models/tech_symbols_{version}.pkl', 'rb') as f:
        symbols = pickle.load(f)
    max_look_back = max([look_back_1, look_back_2])

    if update:
        get_sector(sector, symbols, start_date='2010-1-1', end_date='2021-3-25')

    dates, close_prices, normalised_close_prices, N_tickers, normaliser, symbols = prepare_data(sector, split=False, start_date='2019-01-01', end_date='2021-03-25', save_model_stocks=False, load_normaliser=True, version=version)

    _, _, y_true = create_dataset(normalised_close_prices, look_back_1, look_back_2, look_forward)

    x_1, x_2, _ = create_dataset(normalised_close_prices, look_back_1, look_back_2, 0)
    dates_true = dates[max_look_back:-look_forward]
    dates = dates[max_look_back:]

    y_real = close_prices[max_look_back:]

    model = load_model(model_name, version)

    y_preds = model.predict([x_1, x_2])

    latest_preds = y_preds[-1]

    positive_weight_total = sum([latest_pred - 0.5 for symbol, latest_pred in zip(symbols, latest_preds) if latest_pred > 0.5 and symbol in purchase_symbols])
    positive_weight_dict = {symbol: (latest_pred - 0.5) / positive_weight_total for symbol, latest_pred in zip(symbols, latest_preds) if latest_pred > 0.5 and symbol in purchase_symbols}

    positive_weight_sqrt_total = sum([np.sqrt(latest_pred - 0.5) for symbol, latest_pred in zip(symbols, latest_preds) if latest_pred > 0.5 and symbol in purchase_symbols])
    positive_weight_dict_sqrt = {symbol: np.sqrt(latest_pred - 0.5) / positive_weight_sqrt_total for symbol, latest_pred in zip(symbols, latest_preds) if latest_pred > 0.5 and symbol in purchase_symbols}

    chosen_positions = {symbol: latest_pred for symbol, latest_pred in zip(symbols, latest_preds) if symbol in chosen_stocks}

    pp.pprint(chosen_positions)

    # correlations = {}
    # for symbol_idx in range(y_preds.shape[1]):
    #     correlation = np.corrcoef(y_preds[:len(y_true), symbol_idx], y_true[:, symbol_idx])[0][1]
    #     correlations[symbols[symbol_idx]] = correlation

    # pp.pprint(correlations)

    earnings, index_earnings, optimal_earnings = [], [], []
    earnings_dict, index_earnings_dict, optimal_earnings_dict = {}, {}, {}
    for target_stock in chosen_stocks:
        target_stock_idx = symbols.index(target_stock)
        print(f"Target Stock: {target_stock}")
        earning, index_earning, optimal_earning = evaluate_earnings(dates, y_preds, dates_true, y_true, dates, y_real, target_stock_idx, target_stock)

        earnings.append(earning)
        index_earnings.append(index_earning)
        optimal_earnings.append(optimal_earning)

        earnings_dict[target_stock] = earning
        index_earnings_dict[target_stock] = index_earning
        optimal_earnings_dict[target_stock] = optimal_earning

    print(f"Average Earnings: {np.mean(earnings)}")
    print(f"Average Index Performance: {np.mean(index_earnings)}")

    # import matplotlib.pyplot as plt
    # for target_stock in symbols:
    #     plt.plot(correlations[target_stock], earnings_dict[target_stock], '.', label=target_stock)
    # plt.legend()
    # plt.xlabel('Correlation')
    # plt.ylabel('Earnings to Index Earnings Ratio')
    # plt.show()
