import matplotlib.pyplot as plt


def evaluate_earnings(dates_pred, y_preds, dates_true, y_true, dates, y_real, target_stock, symbol):
    target_stock_pred = y_preds[:, target_stock]
    target_stock_true = y_true[:, target_stock]
    target_stock_real = y_real[:, target_stock]

    plt.subplot(211)
    plt.plot(dates_pred, target_stock_pred, label='pred')
    plt.plot(dates_true, target_stock_true, label='true')
    plt.legend()
    plt.subplot(212)
    plt.suptitle(symbol)
    plt.plot(dates, target_stock_real)

    wallet, ideal_wallet = 1, 1
    state, ideal_state = 'sell', 'sell'
    for idx, target_stock_prediction in enumerate(target_stock_pred):
        if target_stock_prediction > 0.5 and idx < (len(target_stock_pred) - 1):
            if state != 'buy':
                bought_val = target_stock_real[idx]
                state = 'buy'
                plt.plot(dates[idx], target_stock_real[idx], '.', markersize=20, color='green')
        else:
            if state != 'sell':
                sold_val = target_stock_real[idx]
                delta = (sold_val - bought_val) / bought_val
                wallet = wallet * (1 + delta)
                state = 'sell'
                plt.plot(dates[idx], target_stock_real[idx], '.', markersize=20, color='red')

        if idx < (len(target_stock_true) - 1) and target_stock_true[idx] > 0.5:
            if ideal_state != 'buy':
                ideal_bought_val = target_stock_real[idx]
                ideal_state = 'buy'
        elif idx <= (len(target_stock_true) - 1):
            if ideal_state != 'sell':
                ideal_sold_val = target_stock_real[idx]
                delta = (ideal_sold_val - ideal_bought_val) / ideal_bought_val
                ideal_wallet = ideal_wallet * (1 + delta)
                ideal_state = 'sell'

    print(f"Profit: {round(wallet * 100)}%")
    print(f"Ideal Profit: {round(ideal_wallet * 100)}%")
    print(f"Base profit: {round((target_stock_real[-1] / target_stock_real[0]) * 100)}%\n")

    # plt.show()
    plt.close()

    return wallet, target_stock_real[-1] / target_stock_real[0], ideal_wallet
