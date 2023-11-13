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
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

# Define the critic network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

# Define the DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, tau=0.001):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)

        # Initialize target networks with the same weights as the online networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)

        # Update critic
        target_value = reward + (1 - done) * self.gamma * self.critic_target(next_state, self.actor_target(next_state))
        predicted_value = self.critic(state, action)
        critic_loss = F.mse_loss(predicted_value, target_value)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Update target networks
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

    def soft_update(self, online_model, target_model):
        for online_params, target_params in zip(online_model.parameters(), target_model.parameters()):
            target_params.data.copy_(self.tau * online_params.data + (1.0 - self.tau) * target_params.data)

# Example usage
state_dim = 4
action_dim = 1
agent = DDPGAgent(state_dim, action_dim)

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
