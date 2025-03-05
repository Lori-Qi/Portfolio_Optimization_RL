import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
from tqdm import tqdm
from collections import deque


# DQN Agent
# Implement epsilon-greedy exploration policy

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(
        self,
        state_size = 4,
        action_space = np.linspace(-1,2,31),
        memory_size = 2000,
        gamma = 0.95,
        epsilon = 1,
        epsilon_min = 0.01,
        epsilon_decay = 0.995,
        learning_rate = 0.001,
        batch_size = 32,
        device = None
    ):
        self.state_size = state_size
        self.state_size = state_size
        self.action_space = action_space
        self.action_size = len(action_space)
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # set the device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # create networks
        self.model = DQN(state_size, self.action_size).to(self.device)
        self.target_model = DQN(state_size, self.action_size).to(self.device)
        self.update_target_model()

        # set the optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # track the training progress
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []

        # copy weights from model to target model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # epsilon-greedy action selection
        if np.random.rand() <= self.epsilon:
            action_idx = np.random.randint(self.action_size) # random action
        else:
            # predict Q-values for all actions
            with torch.no_grad():
                self.model.eval()
                q_values = self.model(state)
                action_idx = torch.argmax(q_values).item()
                self.model.train() # ?

        return self.action_space[action_idx], action_idx
        

    def replay(self, batch_size = None):
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.memory) < batch_size:
            return 0

        # sample a batch from memory
        minibatch = random.sample(self.memory, batch_size)
            
        losses = []

        self.optimizer.zero_grad()

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        # prepare batch data
        for state, action_idx, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action_idx)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        # convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        # next_state = torch.FloatTensor(next_states).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # compute current Q values
        self.model.train()
        # current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        current_q_values = self.model(states).gather(1, actions.view(-1, 1))  # actions.view(-1, 1) 改变为列向量
        current_q_values = current_q_values.squeeze(1)  # 去除不必要的维度

        # compute next Q values
        with torch.no_grad():
            self.target_model.eval()
            next_actions = self.model(next_states).max(1)[1]

            next_q_values = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # backpropagation
        loss.backward()
        self.optimizer.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # save the loss
        loss_val = loss.item()
        self.loss_history.append(loss_val)

        return loss_val

    def load(self, name):
        self.model.load_state_dict(torch.load(name))
        self.update_target_model()

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def get_metric(self):
        return{
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'epsilon_history': self.epsilon_history
        }