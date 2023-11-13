import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define the actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.fc2(x), dim=-1)
        return action_probs

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value

# Define the Actor-Critic agent
class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        return action.item()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)

        # Compute TD error
        value = self.critic(state)
        next_value = self.critic(next_state) if not done else 0
        td_error = reward + self.gamma * next_value - value

        # Update critic
        critic_loss = F.smooth_l1_loss(value, reward + self.gamma * next_value)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update actor
        action_probs = self.actor(state)
        action_dist = torch.distributions.Categorical(action_probs)
        actor_loss = -action_dist.log_prob(action) * td_error.detach()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

# Example usage
state_dim = 4
action_dim = 2
agent = ActorCriticAgent(state_dim, action_dim)

# Training loop (replace this with your environment interactions)
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
