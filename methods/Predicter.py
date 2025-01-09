import numpy as np
from env import config


class SMAPredictor(object):
    def __init__(self, ms_nums, window_size):
        self.ms_nums = ms_nums
        self.window_size = window_size
        self.buffer = []
        self.init_lamda = None

    def reset(self, init_lamda):
        self.buffer.clear()
        self.init_lamda = init_lamda

    def record(self, value: list):
        self.buffer.append(value)

    def predict(self):
        if self.buffer == []:
            raise ValueError("Please call record() first!")
        
            """ 废弃 """
            if self.init_lamda is None:
                raise ValueError("Please call reset() first!")
            return [self.init_lamda for _ in range(self.ms_nums)]
        
        elif len(self.buffer) < self.window_size:
            return np.mean(self.buffer, axis=0)
        
        return np.mean(self.buffer[-self.window_size:], axis=0)