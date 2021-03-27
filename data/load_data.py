import json
import requests
import io
import os

import pandas as pd
from scipy.interpolate import interp1d


ENERGY_SYMBOLS = ['XOM', 'CVX', 'RDS-A', 'RDS-B', 'TOT', 'PTR', 'PBR', 'PBR-A', 'BP', 'SNP', 'SLB', 'EPD', 'E', 'COP',
                  'EQNR', 'EOG', 'CEO', 'SU', 'OXY', 'KMI', 'PSX', 'HAL']
TECH_SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'TSLA', 'BABA', 'NVDA', 'PYPL', 'INTC', 'CRM', 'AMD', 'ATVI', 'MTCH',
                'EA', 'ZG', 'TTD', 'YELP', 'ADBE', 'CSCO', 'ASML', 'ORCL', 'AVGO', 'TXN', 'SHOP', 'QCOM', 'SAP', 'SQ', 'INTU',
                'ZM', 'AMAT', 'NOW', 'IBM', 'BIDU', 'TSM']


def get_dataset(symbol, start_date, end_date):
    with open('data/creds.json', 'r') as f:
        credentials = json.load(f)

    url = f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?startDate={start_date}&endDate={end_date}&token={credentials['tiingo_api']}&format=csv"
    response = requests.get(url, headers={'Content-Type': 'application/json'})

    data = pd.read_csv(io.StringIO(response.content.decode('utf-8')))

    if len(data) > 0:
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data = data[['date', 'adjClose']]
        data.rename(columns={'adjClose': symbol}, inplace=True)

        return data
    else:
        return False


def get_sector(sector, symbols=None, start_date='2010-1-1', end_date='2021-2-1'):
    if symbols is None:
        if sector == 'tech':
            symbols = TECH_SYMBOLS
        elif sector == 'energy':
            symbols = ENERGY_SYMBOLS

    master_df = pd.DataFrame(data={'date': pd.date_range(start=start_date, end=end_date, freq='1D')})

    for symbol in symbols:
        df = get_dataset(symbol, start_date, end_date)

        if df is not False:
            master_df = pd.merge_asof(master_df, df, on='date')

    os.makedirs('data', exist_ok=True)
    master_df.to_csv(f'data/{sector}_master.csv', index=False)


if __name__ == "__main__":
    sector = 'tech'
    get_sector(sector)
