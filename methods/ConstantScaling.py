import random
from env import environment
from env import config
import numpy as np


class ConstantScalingAgent(object):
    def __init__(self, env: environment.DataCenterEnvironment):
        self.env = env
        self.actoin_space_dim = np.array([env.server_node_nums,
                                          env.ms_nums,
                                          env.config.max_instance_update_num * 2 + 1])
        pass

    def get_action(self, state):
        """
        不做任何伸缩
        return: np.array([node_id, ms_id, ms_image_nums])
        """
        (node_id, ms_id, ms_image_nums) = (0, 0, 0)

        return np.array([node_id, ms_id, ms_image_nums])
