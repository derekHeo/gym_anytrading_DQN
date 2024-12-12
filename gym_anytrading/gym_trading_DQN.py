import os
import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
import matplotlib.pyplot as plt
from gym_anytrading.envs import Actions, Positions
from datasets import STOCKS_GOOGL, STOCKS_BOEING
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=2):
        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        n_input_features = observation_space.shape[1]

        self.lstm = nn.LSTM(input_size=n_input_features, hidden_size=64, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        lstm_out, _ = self.lstm(observations)
        last_timestep = lstm_out[:, -1, :]
        return self.linear(last_timestep)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -float('inf')

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                mean_reward = sum(y[-100:]) / len(y[-100:])
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"New best mean reward: {mean_reward}. Saving model to {self.save_path}")
        return True

def train(env_id, df, log_base_dir="logs", model_base_dir="models", model_name="anytrading_dqn"):
    log_path = os.path.join(log_base_dir, model_name)
    os.makedirs(log_path, exist_ok=True)

    env = gym.make(env_id, df=df, window_size=10, frame_bound=(10, 300))
    vec_env = make_vec_env(lambda: env, n_envs=1)
    vec_env = VecMonitor(vec_env, log_path)

    policy_kwargs = dict(
        features_extractor_class=CustomLSTMFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=2)
    )

    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=0.001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=128,
        tau=0.01,
        gamma=0.95,
        train_freq=4,
        gradient_steps=4,
        exploration_fraction=0.15,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs
    )

    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_path)
    #여기서 timesteps 지정
    model.learn(total_timesteps=200000, callback=callback)

    # 저장 save
    model_save_path = os.path.join(model_base_dir, model_name)
    os.makedirs(model_base_dir, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

def test(env_id, df, model_base_dir="models", model_name="anytrading_dqn"):
    model_path = os.path.join(model_base_dir, model_name)
    env = gym.make(env_id, df=df, window_size=10, frame_bound=(10, 300), render_mode="human")

    vec_env = make_vec_env(lambda: env, n_envs=1)

    model = DQN.load(model_path, env=vec_env)

    obs = vec_env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += rewards[0]
        if dones[0]:
            print(f"Total Reward: {total_reward}")
            break

    env.unwrapped.render_all(title="Trading Performance")

if __name__ == "__main__":
    env_id = "stocks-v0"
    # train(env_id, STOCKS_GOOGL)
    test(env_id, STOCKS_GOOGL)
