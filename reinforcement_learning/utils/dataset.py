from collections import deque
import random


class Dataset:
    def __init__(self, max_length):
        self._memory = deque(maxlen=max_length)

    def push(self, data):
        self._memory.append(data)

    def sample(self, batch_size):
        return random.sample(self._memory, min(batch_size, len(self._memory)))
