import numpy as np


class Agent:
    def __init__(self, policy):
        self._policy = policy

    def select_action(self, state, epsilon=0.):
        if np.random.uniform() < epsilon:
            return np.random.choice([0, 1])
        distribution = self._policy.get_distribution(state)
        return np.argmax(distribution)
