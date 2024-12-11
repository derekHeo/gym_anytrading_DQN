import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import TradingEnv, ForexEnv, StocksEnv, Actions, Positions
from datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL, STOCKS_BOEING


env = gym.make('stocks',df = STOCKS_BOEING, frame_bound=(50, 100), window_size=10)

observation = env.reset(seed=42)
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # env.render()
    if done:
        print("info:", info)
        break

plt.cla()
env.unwrapped.render_all()
plt.show()