import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
import gym_anytrading
from gym_anytrading.envs import Actions
import matplotlib.pyplot as plt

from datasets import FOREX_EURUSD_1H_ASK, STOCKS_GOOGL, STOCKS_BOEING

# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)  # batch_first=True로 설정
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
        # 입력 데이터의 형태 조정: [batch_size, seq_len, feature_dim]
        x = x.unsqueeze(1) if len(x.shape) == 2 else x  # 시퀀스 차원 추가
        lstm_out, _ = self.lstm(x)  # LSTM 처리
        x = lstm_out[:, -1, :]  # 마지막 타임스텝 출력 사용
        return self.fc(x)


# Replay Buffer 정의
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self):
        return len(self.buffer)

# DQN 에이전트 정의
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.policy_net = DQN(state_dim, action_dim).float()
        self.target_net = DQN(state_dim, action_dim).float()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(max_size=30000)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def train(self, batch_size=128):
        if self.replay_buffer.size() < batch_size * 5:
            return

        # 샘플링된 데이터를 텐서로 변환
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # 상태 데이터를 [batch_size, seq_len, feature_dim] 형태로 맞춤
        states = states.view(batch_size, -1, self.state_dim)
        next_states = next_states.view(batch_size, -1, self.state_dim)

        # Q-값 계산
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# 환경 및 학습 설정
env = gym.make(
    'stocks',
    df=STOCKS_GOOGL,
    window_size=10,
    frame_bound=(10, 300)
)
state_dim = env.observation_space.shape[1] * env.observation_space.shape[0]  # Flattened observation shape
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
episodes = 100
target_update_interval = 10

for episode in range(episodes):
    state, _ = env.reset()
    state = state.flatten() # Flatten the observation
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state.flatten()
        done = terminated or truncated

        agent.replay_buffer.add(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        total_reward += reward

        if done:
            if episode % target_update_interval == 0:
                agent.update_target_net()
            print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
            break

    agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

# 테스트 실행
state, _ = env.reset()
state = state.flatten()
total_reward = 0

while True:
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state.flatten()
    total_reward += reward

    if terminated or truncated:
        print(f"Test Total Reward: {total_reward:.2f}")
        break

plt.cla()
env.unwrapped.render_all()
plt.show()
