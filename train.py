from tabnanny import check

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from gymnasium import Env
from MarketEnvironmentSimulator import GymTradingEnvironment
from MarketEnvironmentSimulator import generate_synthetic_stock_data

price_data = generate_synthetic_stock_data(200)

env = GymTradingEnvironment(price_data)

check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100_000)

model.save("ppo_trading_model")