import random

import gym
from gym import spaces
import numpy as np

MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 100


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df

        self.symbols = [symbol for symbol in list(self.df) if '_grad' not in symbol]

        self.comission = 0.0045

        # Actions of the format Buy x%, Sell x%, Hold
        self.action_space = spaces.Discrete(len(self.symbols) * 2)

        # Prices contains the close price and predicted gradient for the symbols
        self.observation_space = spaces.Discrete(int(self.df.shape[1]))

    def _next_observation(self):
        # Get the current stock data and scale to between 0-1
        obs = self.df.loc[self.current_step].to_numpy()

        obs = obs[len(self.symbols):]

        n_shares_all = []
        for symbol in self.symbols:
            if symbol in self.shares_held.keys():
                n_shares = self.shares_held[symbol]
            else:
                n_shares = 0

            n_shares_all.append(n_shares)

        obs = np.append(obs, n_shares_all)

        return obs

    def _take_action(self, action):
        self.net_worth = 0

        for idx, symbol in enumerate(self.symbols):
            action_type = action[idx]
            amount = action[idx + len(self.symbols)]

            share_price = self.df.loc[self.current_step, symbol]

            self.shares_held.setdefault(symbol, 0)

            if action_type < (1 / 3):
                # Sell amount % of shares held
                sell_value = amount * self.shares_held[symbol] * share_price

                self.balance += sell_value

                self.shares_held[symbol] -= amount * self.shares_held[symbol] * (1 - self.comission)

            elif action_type > (2 / 3):
                # Buy amount % of balance in shares
                buy_cost = amount * self.balance

                self.balance -= buy_cost

                self.shares_held[symbol] += (buy_cost * (1 - self.comission)) / share_price

            self.net_worth += self.shares_held[symbol] * share_price

        self.net_worth += self.balance

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step >= self.df.shape[0]:
            self.current_step = 0

        reward = self.net_worth

        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = {}

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(0, self.df.shape[0] - 1)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth / INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth}')
        print(f'Profit: {profit}')
