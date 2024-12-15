from datastruct import *
from config import EnvConfig
from environment import DataCenterEnvironment
import sys
sys.path.append('./')
from methods import FFD
import numpy as np
import copy


config = EnvConfig()
random.seed(1037)
np.random.seed(1037)

def _test_ms_instance():
    ms_instance = MSInstance(
                            0,
                            config.ms_min_cpu_resource,
                            config.ms_max_cpu_resource,
                            config.ms_min_memory_resource,
                            config.ms_max_memory_resource
                            )
    print(ms_instance.get_cpu())
    print(ms_instance.get_memory())
    print("success")

def _test_server_node():
    node = Node(
                0,
                config.node_min_compute_speed,
                config.node_max_compute_speed,
                config.node_min_cpu_resource,
                config.node_max_cpu_resource,
                config.node_min_memory_resource,
                config.node_max_memory_resource
                )
    ms_instance = MSInstance(
                            0,
                            config.ms_min_cpu_resource,
                            config.ms_max_cpu_resource,
                            config.ms_min_memory_resource,
                            config.ms_max_memory_resource
                            )
    flag1 = node.is_resource_enough(ms_instance, 1)
    flag2 = node.is_resource_enough(ms_instance, -1)
    node.delpoy(ms_instance, 1)
    flag3 = node.is_resource_enough(ms_instance, -1)
    # flag4 = node.delpoy(ms_instance, -2)
    print("success")

def _test_init_deploy():
    ms_num = 10
    node_num = 5
    ms_list = [MSInstance(i, 1, 2, 10, 20) for i in range(ms_num)]
    ms_image_list = [4 for _ in range(ms_num)]
    node_list = [Node(i, 1, 5, 10, 20, 100, 200) for i in range(node_num)]
    state = np.zeros((3, ms_num, node_num))
    for node_idx in range(state[0].shape[1]):
            node = node_list[node_idx]
            state[1][:, node_idx] = node.get_cpu()
            state[2][:, node_idx] = node.get_memory()
    init_deploy_method = FFD.FFD(ms_list, ms_image_list, node_list, state)
    new_state, new_node_list = init_deploy_method.deploy()

    print("success")

def _test_env():
    env = DataCenterEnvironment(config)
    env.reset()
    action1 = np.zeros(env.state[0].shape, dtype=int)
    state1, reward1, _, _ = env.step(action1)
    action2 = copy.deepcopy(action1)
    action2[0][0] = -2
    action2[0][4] = 2
    state2, reward2, _, _ = env.step(action2)
    print("success")


if __name__ == '__main__':
    # _test_ms_instance()
    # _test_server_node()
    # _test_init_deploy()
    _test_env()