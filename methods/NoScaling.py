import random
from env import environment
from env import config
import numpy as np


class NoScalingAgent(object):
    def __init__(self, env: environment.DataCenterEnvironment):
        self.env = env  # 环境的引用，实时监控env运行过程的状态变化
        self.actoin_space_dim = np.array([env.server_node_nums,
                                          env.ms_nums,
                                          env.config.max_instance_update_num * 2 + 1], dtype=int)
        pass

    def get_action(self, state):
        """
        不做任何伸缩
        return: np.array([node_id, ms_id, ms_image_nums])
        """
        (node_id, ms_id, ms_image_nums) = (0, 0, (self.actoin_space_dim[2]-1)/2)

        return np.array([node_id, ms_id, ms_image_nums], dtype=int)
