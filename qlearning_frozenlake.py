from pprint import pprint

import gymnasium as gym
import numpy as np

class QLearning():

    def __init__(self, states, n_actions, gamma, alpha):
        self.Q = {state: [0, ] * n_actions for state in states}
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha

    def learn(self, s, a, r, s_):
        self.Q[s][a] = self.Q[s][a] + self.alpha * (r + self.gamma * max(self.Q[s_]) - self.Q[s][a])

    def decide(self, s):
        return np.argmax(self.Q[s])

def main():
    env = gym.make('FrozenLake-v1', desc=None, map_name='4x4', is_slippery=False, render_mode='human')
    agent = QLearning(range(env.observation_space.n), env.action_space.n, 1, 0.5)
    state, info = env.reset()
    for _ in range(1000):
        # action = env.action_space.sample()  # TODO: Replace this
        action = agent.decide(state)
        new_state, reward, terminated, truncated, info = env.step(action)
        # Penalize the agent for each step taken so that it tries to find the fastest solution
        if terminated:
            if reward == 1:
                reward = 10
            else:
                reward = -10
        else:
            reward = -1
        agent.learn(state, action, reward, new_state)
        print(state, action, reward, new_state, terminated, truncated, info)
        pprint(agent.Q)
        if terminated or truncated:
            state, info = env.reset()
        else:
            state = new_state
    env.close()

if __name__ == "__main__":
    main()
