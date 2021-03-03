import json
import requests
import io
import os

import pandas as pd
from scipy.interpolate import interp1d


ENERGY_SYMBOLS = ['XOM', 'CVX', 'RDS-A', 'RDS-B', 'TOT', 'PTR', 'PBR', 'PBR-A', 'BP', 'SNP', 'SLB', 'EPD', 'E', 'COP',
                  'EQNR', 'EOG', 'CEO', 'SU', 'OXY', 'KMI', 'PSX', 'HAL']
TECH_SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'BABA', 'NVDA', 'PYPL', 'INTC', 'CRM', 'AMD', 'ATVI', 'MTCH',
                'EA', 'ZG', 'TTD', 'YELP', 'ADBE', 'CRM', 'CSCO', 'TSM', 'ASML', 'ORCL', 'AVGO', 'TXN', 'SHOP',
                'QCOM', 'SAP', 'SQ', 'INTU', 'ZM', 'AMAT', 'NOW', 'IBM', 'BIDU']


def save_dataset(symbol, sector):
    with open('data/creds.json', 'r') as f:
        credentials = json.load(f)

    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate=2011-1-1&endDate=2019-1-1&token={credentials['tiingo_api']}&format=csv"
    response = requests.get(url, headers={'Content-Type': 'application/json'})

    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

    if len(data) > 0:
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data['date'] = (data['date'] - data['date'].iloc[0]).dt.days

        x = data['date'].values
        y = data['close'].values

        f = interp1d(x, y)

        xnew = range(x[0], x[-1])

        y = f(xnew)

        data = pd.Series(index=xnew, data=f(xnew))

        os.makedirs('data', exist_ok=True)
        data.to_csv(f'data/{sector}/{symbol}_close.csv', header=False, index=False)


def get_sector(sector):
    if sector == 'tech':
        symbols = TECH_SYMBOLS
    elif sector == 'energy':
        symbols = ENERGY_SYMBOLS

    existing_symbols = list(filter(lambda x: x.split('_')[0], os.listdir(f'data/{sector}')))

    for symbol in symbols:
        if symbol not in existing_symbols:
            save_dataset(symbol, sector)


if __name__ == "__main__":
    sector = 'tech'
    get_sector(sector)
