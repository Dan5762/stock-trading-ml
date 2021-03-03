import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.diverging import RdYlGn_4


def evaluate_earnings_classifier(x_test, y_test, model, target_stock, real_price):
    target_real_price = real_price[:, target_stock]

    test_data_predictions = model.predict(x_test)
    target_stock_predictions = test_data_predictions[:, target_stock]
    target_stock_true = y_test[:, target_stock]

    plt.figure(1)
    plt.subplot(211)
    plt.plot(target_real_price)
    plt.subplot(212)
    plt.plot(target_stock_predictions, label='pred')
    plt.plot(target_stock_true, label='true')
    plt.legend()

    xy = np.array([np.arange(0, len(target_real_price), step=1), target_real_price]).T.reshape(-1, 1, 2)
    segments = np.hstack([xy[:-1], xy[1:]])

    plt.figure(2)

    cmap = RdYlGn_4.mpl_colormap

    for idx, segment in enumerate(segments):
        color = cmap(target_stock_predictions[idx])
        plt.plot(segment.T[0], segment.T[1], c=color)

    wallet = 1
    state = 'sell'
    for idx, target_stock_prediction in enumerate(target_stock_predictions):
        if target_stock_prediction > 0.5 and idx < (len(target_stock_predictions) - 1):
            if state != 'buy':
                bought_val = target_real_price[idx]
                state = 'buy'
                plt.plot(idx, target_real_price[idx], '.', markersize=20, color='green')
        else:
            if state != 'sell':
                sold_val = target_real_price[idx]
                delta = (sold_val - bought_val) / bought_val
                wallet = wallet * (1 + delta)
                state = 'sell'
                plt.plot(idx, target_real_price[idx], '.', markersize=20, color='red')

    print(f"Profit: {round(wallet * 100)}%")
    print(f"Base profit: {round((target_real_price[-1] / target_real_price[0]) * 100)}%\n")

    # plt.figure(1)
    # plt.show()
    plt.close(1)

    # plt.figure(2)
    # plt.show()
    plt.close(2)

    return wallet, target_real_price[-1] / target_real_price[0]
