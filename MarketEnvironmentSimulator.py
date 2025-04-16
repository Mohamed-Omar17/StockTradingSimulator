import numpy as np
import pandas as pd
import random
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
from sympy.physics.units import current

x = []
y = []

#Set up the stock price data source

def generate_synthetic_stock_data(days=100, start_price=100):
    prices = [start_price] #starting price
    for _ in range(1, days):
        prices.append(prices[-1] * (1+np.random.normal(0,0.01)))
    return pd.DataFrame({'price': prices})


class MarketState:
    def __init__(self, price_series):
        self.price_series = price_series
        self.current_step = 0
        self.current_price = self.price_series[0]

    def step(self):
        self.current_step += 1
        if self.current_step < len(self.price_series):
            self.current_price = self.price_series[self.current_step]
        else:
            raise IndexError("End of price series reached.")

class Portfolio:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.shares = 0

    def buy(self, price):
        if self.cash >= price:
            self.shares += 1
            self.cash -= price

    def sell(self, price):
        if self.shares > 0:
            self.shares -= 1
            self.cash += price

    def get_value(self, price):
        return self.cash + self.shares * price

class GymTradingEnvironment(gym.Env):
    def __init__(self, data):
        super(GymTradingEnvironment, self).__init__()
        self.current_step = 0
        self.price_data = data['price'].values
        self.initial_cash = 1000
        self.portfolio = Portfolio(self.initial_cash)

        #Define actions 0, 1, 2
        self.action_space = spaces.Discrete(3)

        #Observation space: [price, cash, shares]

        obs_low = np.array([0,0,0], dtype=np.float64)
        obs_high = np.array([1e6, 1e6, 1e6])
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float64)

        self.reset()


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.state = MarketState(self.price_data)
        self.portfolio = Portfolio(self.initial_cash)
        self.done = False
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        if self.done:
            raise Exception("Simulation already finished.")

        price = self.state.current_price
        current_day = self.current_step

        if action == 1:
            self.portfolio.buy(price)
        elif action == 2:
            self.portfolio.sell(price)

        try:
            self.state.step()
        except IndexError:
            self.done = True

        new_price = self.state.current_price
        reward = self.portfolio.get_value(self.state.current_price)
        observation = self._get_observation()
        done = self.state.current_step >= len(self.price_data - 1)

        x.append(current_day)
        y.append(price)

        self.current_step+=1




        return observation, reward, self.done, False, {}

    def _get_observation(self):
        return np.array([
            self.state.current_price,
            self.portfolio.cash,
            self.portfolio.shares
        ], dtype=np.float64)


price_data = generate_synthetic_stock_data(200)
env = GymTradingEnvironment(price_data)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs, _ = env.reset()
done = False

while not done:
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(f"Action: {['HOLD', 'BUY', 'SELL'][int(action)]}, Portfolio Value: ${reward:.2f}")


plt.plot(x, y)
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.title("Stock Price over time")

final_value = env.portfolio.get_value(env.state.current_price)
print(f"Final Portfolio Value: ${final_value:.2f}")

plt.show()