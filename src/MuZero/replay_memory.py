import numpy as np

class ReplayBuffer:

    def __init__(self, config):

        self.buffer = []
        self.buffer_size = int(config['replay_buffer']['buffer_size'])
        self.sample_size = int(config['replay_buffer']['sample_size'])

    def add(self, game):

        if len(self.buffer) >= self.buffer_size: self.buffer.pop(0)
        self.buffer.append(game)

    def sample(self):

        if len(self.buffer) <= self.sample_size:
            return self.buffer.copy()
        return np.random.choice(self.buffer, size=self.sample_size, replace=False).tolist()