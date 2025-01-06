import numpy as np
from env import config


class SMAPredictor(object):
    def __init__(self, ms_nums, window_size):
        self.config = config.EnvConfig()
        self.ms_nums = ms_nums
        self.window_size = window_size
        self.buffer = []
        self.init_lamda = self.config.init_lamda

    def reset(self):
        self.buffer.clear()

    def record(self, value: list):
        self.buffer.append(value)

    def predict(self):
        if self.buffer == []:
            return [self.init_lamda for _ in range(self.ms_nums)]
        
        elif len(self.buffer) < self.window_size:
            return np.mean(self.buffer, axis=0)
        
        return np.mean(self.buffer[-self.window_size:], axis=0)