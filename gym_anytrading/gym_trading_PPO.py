import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
from torch import nn
import matplotlib.pyplot as plt
from gym_anytrading.envs import Actions, Positions
from datasets import STOCKS_GOOGL, STOCKS_BOEING
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.policies import ActorCriticPolicy

# Custom feature extractor for LSTM
class CustomLSTMFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(CustomLSTMFeaturesExtractor, self).__init__(observation_space, features_dim)
        n_input_features = observation_space.shape[1]

        self.lstm = nn.LSTM(input_size=n_input_features, hidden_size=32, batch_first=True)  # Reduced hidden size
        self.linear = nn.Sequential(
            nn.Linear(32, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        lstm_out, _ = self.lstm(observations)
        last_timestep = lstm_out[:, -1, :]
        return self.linear(last_timestep)
# Callback to save the best model
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

# Training function
def train(env_id, df, log_base_dir="logs", model_base_dir="models", model_name="anytrading_ppo"):
    log_path = os.path.join(log_base_dir, model_name)
    os.makedirs(log_path, exist_ok=True)

    # Create the environment
    env = gym.make(env_id, df=df, window_size=10, frame_bound=(10, 300))
    vec_env = make_vec_env(lambda: env, n_envs=1)
    vec_env = VecMonitor(vec_env, log_path)

    # Define custom policy with LSTM feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomLSTMFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128)
    )

    # Initialize the model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=0.0003,
        n_steps=1024,  # Decreased for faster updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,  # Increased to encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,  # Set to 1 for better debugging
        tensorboard_log=log_path,
        policy_kwargs=policy_kwargs
    )

    # Train the model
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_path)
    model.learn(total_timesteps=100000, callback=callback)

    # Save the final model
    model_save_path = os.path.join(model_base_dir, model_name)
    os.makedirs(model_base_dir, exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

# Testing function
def test(env_id, df, model_base_dir="models", model_name="anytrading_ppo"):
    model_path = os.path.join(model_base_dir, model_name)
    env = gym.make(env_id, df=df, window_size=10, frame_bound=(10, 300), render_mode="human")

    # Wrap the environment with VecMonitor for compatibility
    vec_env = make_vec_env(lambda: env, n_envs=1)

    # Load the trained model
    model = PPO.load(model_path, env=vec_env)

    # Test the model
    obs = vec_env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += rewards[0]
        if dones[0]:
            print(f"Total Reward: {total_reward}")
            break

    # Render the entire trading performance
    env.unwrapped.render_all(title="Trading Performance")

if __name__ == "__main__":
    env_id = "stocks-v0"
    train(env_id, STOCKS_GOOGL)
    test(env_id, STOCKS_GOOGL)
