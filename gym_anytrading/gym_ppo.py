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

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.common = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)  # 행동 확률 출력
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 상태 가치 평가
        )

    def forward(self, state):
        common_features = self.common(state)
        action_probs = self.actor(common_features)
        state_value = self.critic(common_features)
        return action_probs, state_value

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, clip_ratio=0.2, k_epochs=4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.k_epochs = k_epochs

        self.policy = ActorCritic(state_dim, action_dim).float()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.memory = []  # 경험 저장

    def store_transition(self, transition):
        self.memory.append(transition)

    def compute_returns(self, rewards, dones, next_value):
        returns = []
        R = next_value if not dones[-1] else 0  # 환경 종료 시 다음 상태 값 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - done)
            returns.insert(0, R)
        return returns

    def train(self):
        # None 값 필터링
        filtered_memory = [m for m in self.memory if m[1] is not None]
        states, actions, log_probs, rewards, dones, next_value = zip(*filtered_memory)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Advantage 및 Returns 계산
        returns = self.compute_returns(rewards, dones, next_value[-1])
        returns = torch.tensor(returns, dtype=torch.float32)
        values = self.policy(states)[1].squeeze()
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO 업데이트
        for _ in range(self.k_epochs):
            new_action_probs, new_values = self.policy(states)
            new_log_probs = torch.log(new_action_probs.gather(1, actions.unsqueeze(1)).squeeze())
            ratios = torch.exp(new_log_probs - log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.memory = []  # 메모리 초기화


# DQN 모델 정의
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),  # 뉴런 수 증가
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )

    def forward(self, x):
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
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.1, epsilon_decay=0.999, epsilon_min=0.01):
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

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

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
    df=STOCKS_BOEING,
    window_size=10,
    frame_bound=(10, 300)
)
state_dim = env.observation_space.shape[1] * env.observation_space.shape[0]
action_dim = env.action_space.n

agent = PPOAgent(state_dim, action_dim)
episodes = 500
batch_size = 128

for episode in range(episodes):
    state, _ = env.reset()
    state = state.flatten()
    total_reward = 0
    rewards, dones = [], []

    while True:
        # 행동 선택
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_probs, value = agent.policy(state_tensor)
        action = torch.multinomial(action_probs, 1).item()
        log_prob = torch.log(action_probs.squeeze(0)[action])

        # 환경 상호작용
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = next_state.flatten()
        done = terminated or truncated

        # 경험 저장
        agent.store_transition((state, action, log_prob, reward, done, value.item()))

        state = next_state
        total_reward += reward
        rewards.append(reward)
        dones.append(done)

        if done:
            # 에피소드 종료 후 학습
            next_value = agent.policy(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))[1].item()
            agent.store_transition((state, None, None, 0, 1, next_value))  # 마지막 상태 추가
            agent.train()
            print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward:.2f}")
            break

# 테스트 실행
state, _ = env.reset()
state = state.flatten()
total_reward = 0

while True:
    # 상태를 텐서로 변환하여 Actor 네트워크를 통해 행동 확률 계산
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    action_probs, _ = agent.policy(state_tensor)

    # 행동을 확률적으로 샘플링
    action = torch.multinomial(action_probs, 1).item()

    # 환경과 상호작용
    next_state, reward, terminated, truncated, info = env.step(action)
    state = next_state.flatten()
    total_reward += reward

    # 에피소드 종료 조건 확인
    if terminated or truncated:
        print(f"Test Total Reward: {total_reward:.2f}")
        break

# 결과 시각화
plt.cla()
env.unwrapped.render_all()
plt.show()

