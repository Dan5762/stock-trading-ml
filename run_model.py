import numpy as np

from data.data_prep import prepare_data
from basic_model import create_dataset, load_model
from util import evaluate_earnings


if __name__ == "__main__":
    sector = 'tech'
    model_name = 'tech_lstm'
    look_back_1 = 40
    look_back_2 = 80
    look_forward = 10
    max_look_back = max([look_back_1, look_back_2])

    dates, close_prices, normalised_close_prices, N_tickers, normaliser, symbols = prepare_data(sector, split=False, start_date='2019-01-01', end_date='2021-02-01', save_model_stocks=False)

    x_1, x_2, y = create_dataset(normalised_close_prices, look_back_1, look_back_2, look_forward)
    dates = dates[max_look_back:]
    y_test_real = close_prices[max_look_back:]

    model = load_model(model_name)

    y_pred = model.predict([x_1, x_2])

    earnings, index_earnings = [], []
    for target_stock in range(N_tickers):
        print(f"Target Stock: {symbols[target_stock]}")
        earning, index_earning = evaluate_earnings(dates, y_pred, dates, y, dates, y_test_real, target_stock, symbols[target_stock])
        earnings.append(earning)
        index_earnings.append(index_earning)

    import matplotlib.pyplot as plt
    plt.hist(earnings, bins=30, label='Algorithm')
    plt.hist(index_earnings, bins=30, label='Index')
    plt.legend()
    plt.show()

    print(f"Average Earnings: {np.mean(earnings)}")
    print(f"Average Index Performance: {np.mean(index_earnings)}")
