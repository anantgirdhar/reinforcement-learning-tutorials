import pickle
# from pprint import pprint
import sys
from time import sleep
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        # Create the layers for a fully connected NN
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cpu')
        self.to(self.device)

    def forward(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # We don't need to activate the last layer because we want the raw values
        # representing the Q values for the various actions
        actions = self.fc3(x)
        return actions

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_ctr = 0
        # Create the DQN used by the agent
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims=256, fc2_dims=256)
        # Initialize the Agent's memory
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.int32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_ctr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_ctr += 1

    def choose_action(self, observation):
        # Use an epsilon-greedy strategy to sample actions
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        # If we don't have enough memories stored up yet, don't bother learning
        if self.mem_ctr < self.batch_size:
            return
        # Zero out the gradients as required by pytorch
        self.Q_eval.optimizer.zero_grad()
        # Figure out which memories we're going to use in this batch
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # Extract the memories and cast them as pytorch tensors
        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        # Get the values of the actions we actually took
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        # Learn
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save(self):
        torch.save(self.Q_eval.state_dict(), 'dql_lunarlander.model')
        with open('dql_lunarlander.p', 'wb') as pfile:
            pickle.dump((self.state_memory, self.new_state_memory, self.action_memory, self.reward_memory, self.terminal_memory, self.epsilon, self.mem_ctr), pfile)

    def load(self):
        if os.path.exists('dql_lunarlander.model') and os.path.exists('dql_lunarlander.p'):
            self.Q_eval.load_state_dict(torch.load('dql_lunarlander.model'))
            with open('dql_lunarlander.p', 'rb') as pfile:
                self.state_memory, self.new_state_memory, self.action_memory, self.reward_memory, self.terminal_memory, self.epsilon, self.mem_ctr = pickle.load(pfile)
                if self.epsilon < self.eps_min:
                    self.epsilon = 0.2
                    self.eps_min = 0.02
                    self.eps_dec = 1e-5

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label='1')
    ax2 = fig.add_subplot(111, label='2', frame_on=False)
    ax.plot(x, epsilons, color='C0')
    ax.set_xlabel('Training Steps', color='C0')
    ax.set_ylabel('Epsilon', color='C0')

    ax.tick_params(axis='x', colors='C0')
    ax.tick_params(axis='y', colors='C0')
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])
    ax2.scatter(x, running_avg, color='C1')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color='C1')
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors='C1')
    if lines is not None:
        for line in lines:
            ptl.axvline(x=line)
    plt.savefig(filename)

def main(show=False, n_games=20000, continuous=False):
    if show:
        env = gym.make('LunarLander-v2', continuous=continuous, render_mode='human')
    else:
        env = gym.make('LunarLander-v2', continuous=continuous)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=256, n_actions=4, eps_end=0.001, eps_dec=1e-6, input_dims=[8], lr=0.003)
    agent.load()
    scores, eps_history = [], []
    try:
        for i in range(n_games):
            score = 0
            done = False
            observation, info = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, truncated, info = env.step(action)
                # print(f'  > {agent.mem_ctr} {observation}, {action}, {reward}, {observation_}, {done}, {agent.epsilon}')
                score += reward
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
                observation = observation_
            scores.append(score)
            eps_history.append(agent.epsilon)
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.2f ' % score, 'average score %.2f ' % avg_score, 'epsilon %.3f' % agent.epsilon, ' last reward ', reward)
    except KeyboardInterrupt:
        print('Killed by user.')
    agent.save()
    x = [i+1 for i in range(len(scores))]
    filename = 'lunarlander.png'
    plot_learning_curve(x, scores, eps_history, filename)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_graphics = sys.argv[1]
        if render_graphics.lower() == 'true':
            main(show=True, n_games=25)
        else:
            main(n_games=5000000)
