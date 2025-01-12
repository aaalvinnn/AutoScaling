import random
from env import environment
import numpy as np


class RandomScalingAgent(object):
    def __init__(self, env: environment.DataCenterEnvironment):
        self.env = env  # 环境的引用，实时监控env运行过程的状态变化
        self.actoin_space_dim = np.array([env.server_node_nums, env.ms_nums, env.config.max_instance_update_num*2+1], dtype=int)
        self.offset = self.env.config.max_instance_update_num
        pass

    def get_action(self, state):
        """
        有条件的随机选择一个action
        return: np.array([node_id, ms_id, delta])
        """
        done = False
        (node_id, ms_id, delta) = (0, 0, 0)
        while not done:
            # 随机抽样
            node_id = random.randint(0, self.actoin_space_dim[0] - 1)
            ms_id = random.randint(0, self.actoin_space_dim[1] - 1)
            delta = random.randint(0, self.actoin_space_dim[2] - 1)
            # 计算动作执行后的服务强度
            lamda = state[3][ms_id][0]
            mu = self.env.MS_list[ms_id].mu * (self.env.ms_image_list[ms_id]+delta-self.offset)
            ro = lamda / (mu + 1e-6)

            # 若节点没有这个实例or资源不足
            node = self.env.Node_list[node_id]
            ms = self.env.MS_list[ms_id]
            if not node.is_resource_enough(ms, delta-self.offset):
                done = False
            # 若弹性伸缩后该微服务队列拥塞
            elif ro >= 1:
                done = False
            else:
                done = True
            
        return np.array([node_id, ms_id, delta])
