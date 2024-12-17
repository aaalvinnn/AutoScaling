import random
from env import environment
from env import config
import numpy as np


class RandomScalingAgent(object):
    def __init__(self, env: environment.DataCenterEnvironment):
        self.env = env
        self.actoin_space_dim = np.array([env.server_node_nums, env.ms_nums, env.config.max_instance_update_num])
        pass

    def get_action(self, state):
        """
        有条件的随机选择一个action
        return: np.array([node_id, ms_id, ms_image_nums])
        """
        done = False
        (node_id, ms_id, ms_image_nums) = (0, 0, 0)
        while not done:
            node_id = random.randint(0, self.actoin_space_dim[0] - 1)
            ms_id = random.randint(0, self.actoin_space_dim[1] - 1)
            ms_image_nums = random.randint(-self.actoin_space_dim[2], self.actoin_space_dim[2])
            # 若不是 卸载掉了一个用户需要且数据中心仅有最后1个实例的微服务
            if not (self.env._is_ms_needed(ms_id) and self.env.ms_image_list[ms_id] <= -ms_image_nums):
                done = True
            
        return np.array([node_id, ms_id, ms_image_nums])
