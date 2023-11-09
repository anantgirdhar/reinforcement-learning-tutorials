import pickle
from pprint import pprint
from time import sleep
import os

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

    def decide(self, s, mask=None):
        if mask is not None:
            values = np.array(self.Q[s] * mask)
            min_value = min(values) - 1
            values += (1 - mask) * min_value
        else:
            values = self.Q[s]
        # print(self.Q[s])
        # print(mask)
        # print(values)
        return np.argmax(values)

    def save_model(self, filename="model_qlearning.p"):
        with open(filename, 'wb') as pfile:
            pickle.dump(self.Q, pfile)

    def load_model(self, filename="model_qlearning.p"):
        if os.path.exists(filename):
            with open(filename, 'rb') as pfile:
                self.Q = pickle.load(pfile)

STATES = {(((r*5+c)*5+p)*4+d): (r, c, p, d) for d in range(4) for p in range(5) for c in range(5) for r in range(5)}

def main():
    env = gym.make('Taxi-v3', render_mode='human')
    # env = gym.make('Taxi-v3')
    agent = QLearning(range(env.observation_space.n), env.action_space.n, 0.99, 0.02)
    agent.load_model()
    state, info = env.reset()
    try:
        for _ in range(50000000):
            action = agent.decide(state, info['action_mask'])
            new_state, reward, terminated, truncated, info = env.step(action)
            # if reward == -1:
            #     reward = -5
            # if new_state == state:
            #     reward = -5
            # elif state in [16, 418, 479, 97] and action != 5:
            #     reward = -8
            # elif state not in [16, 418, 479, 97] and action == 5:
            #     reward = -20
            # elif state in [0, 1, 2, 3, 408, 409, 410, 411, 472, 473, 474, 475, 84, 85, 86, 87]:
            #     if action == 4:
            #         reward = 10
            #     else:
            #         reward = -8
            # elif new_state in [0, 1, 2, 3, 408, 409, 410, 411, 472, 473, 474, 475, 84, 85, 86, 87]:
            #     reward = 5
            agent.learn(state, action, reward, new_state)
            print(STATES[state], action, reward, new_state, terminated, truncated, info)
            # if reward > 0:
                # sleep(1)
            if terminated or truncated:
                state, info = env.reset()
            else:
                state = new_state
    except KeyboardInterrupt:
        pass
    env.close()
    agent.save_model()
    print('Model saved.')

if __name__ == "__main__":
    main()
