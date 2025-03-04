import gym
import mujoco_py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import numpy as np
import matplotlib.pyplot as plt
import copy

env = gym.make('Hopper-v3')

Na = env.action_space.shape[0]
A_MAX = env.action_space.high[0]
A_MIN = env.action_space.low[0]
Ns = env.observation_space.shape[0]
EPISODE = 5000
BUFFER_SIZE = 1e5
BATCH_SIZE = 256
GAMMA = 0.99
LR_C = 1e-3
LR_A = 1e-4
TAU = 1e-3
SIGMA = 0.02

def plot_durations(episode_index, episode_return):
    plt.figure(1)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(episode_index, episode_return)
    plt.pause(0.001)  # pause a bit so that plots are updated

class ReplayBuffer:
    def __init__(self, BUFFER_SIZE):
        self.buffer_size = BUFFER_SIZE
        self.memory = []

    def push(self, data):
        self.memory.append(data)
        if len(self.memory) > self.buffer_size:
            del self.memory[0]

    def sample(self, BATCH_SIZE):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)

class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(Ns, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, Na)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(Ns + Na, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc4 = nn.Linear(Ns + Na, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)

    def forward(self, xs):
        x, a = xs
        x = F.relu(self.fc1(torch.cat([x, a], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.reshape([-1])
        return x

actor_target_net = ActorNet()
actor_net = ActorNet()  # copy.deepcopy(actor_target_net)
critic_target_net = CriticNet()
critic_net = CriticNet()  # copy.deepcopy(critic_target_net)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

optimizer_critic = optim.Adam(critic_net.parameters(), lr=LR_C)  
optimizer_actor = optim.Adam(actor_net.parameters(), lr=LR_A)

def select_action(obs, noiseless=False):
    # TODO: pick action according to actor_net and Gaussian noise
    # If noiseless=True, do not add noise (for evaluation purpose) 
    action = np.random.normal(SIGMA, size=(Na,))
    return torch.tensor(action, dtype=torch.float)

def train():
    if len(replay_buffer) < BATCH_SIZE:
        return
    else:
        sample_batch = replay_buffer.sample(BATCH_SIZE)
    s, a, r, _s, D = zip(*sample_batch)
    state_batch = torch.stack(s)
    action_batch = torch.stack(a)
    reward_batch = torch.tensor(r, dtype=torch.float32)
    _state_batch = torch.stack(_s)
    done_batch = torch.tensor(D, dtype=torch.float32)

    # TODO: calculate critic_loss and perform gradient update
    # optimizer_critic.zero_grad()
    # critic_loss.backward()
    # optimizer_critic.step()

    # TODO: calculate actor_loss and perform gradient update
    # optimizer_actor.zero_grad()
    # actor_loss.backward()
    # optimizer_actor.step()

    # TODO: update target networks

timer = 0
R = 0
Return = []
episode_indexes = []

for episode in range(EPISODE):
    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float)
    done = False
    timer = 0
    R = 0
    eval = (episode % 10 == 0)

    while not done:
        action = select_action(obs, eval)
        action = torch.clamp(action, min=A_MIN, max=A_MAX)
        obs_, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated
        obs_ = torch.tensor(obs_, dtype=torch.float)
        transition = (obs, action, reward, obs_, done)
        replay_buffer.push(transition)
        train()
        R += reward
        timer += 1
        obs = obs_

    if eval:  # evaluation without noise
        print('Episode: %3d,\tStep: %5d,\tReturn: %f' %(episode, timer, R))
        Return.append(R)
        episode_indexes.append(episode)
        plot_durations(episode_indexes, Return)

plt.show()
