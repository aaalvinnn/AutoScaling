from env.datastruct import *
from env.config import EnvConfig
from env.environment import DataCenterEnvironment
from methods import FFD
import numpy as np
import copy


config = EnvConfig()
def reset_seed(seed=1037):        
    random.seed(seed)
    np.random.seed(seed)

reset_seed()

def _test_ms_instance():
    ms_instance = MSInstance(
                            0,
                            config.ms_min_cpu_resource,
                            config.ms_max_cpu_resource,
                            config.ms_min_memory_resource,
                            config.ms_max_memory_resource
                            )
    print(ms_instance.cpu)
    print(ms_instance.memory)
    print("success")

def _test_server_node():
    node = Node(0)
    ms_instance = MSInstance(0)
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
            state[1][:, node_idx] = node.cpu
            state[2][:, node_idx] = node.memory
    init_deploy_method = FFD.FFD(ms_list, ms_image_list, node_list, state)
    new_state, new_node_list = init_deploy_method.deploy()

    print("success")

def _test_env():
    # request chain: [7, 5, 6]
    env = DataCenterEnvironment(config)
    env.reset()
    action1 = np.zeros(env.state[0].shape, dtype=int)
    state1, delay1, _, _ = env.step(action1)

    env.reset()
    action2 = copy.deepcopy(action1)
    action2[5][0] = -1
    action2[5][4] = 1
    state2, delay2, _, _ = env.step(action2)

    env.reset()
    action3 = copy.deepcopy(action1)
    action3[5][0] = -2
    action3[5][4] = 2
    action3[6][0] = -2
    action3[6][3] = 2
    state3, delay3, _, _ = env.step(action3)

    # 路由时延比3更小，因为更多的实例镜像位于相同节点上
    env.reset()
    action4 = copy.deepcopy(action1)
    action4[7][0] = -1
    action4[7][1] = 1
    action4[5][0] = -2
    action4[5][1] = 1
    action4[5][2] = 1
    action4[6][0] = -2
    action4[6][3] = 1
    action4[6][4] = 1
    state4, delay4, _, _ = env.step(action4)

    # 路由时延比4更小，因为更多的实例镜像位于相同节点上
    env.reset()
    action5 = copy.deepcopy(action1)
    action5[7][0] = -1
    action5[7][1] = 1
    action5[5][0] = -2
    action5[5][1] = 1
    action5[5][2] = 1
    action5[6][0] = -2
    action5[6][2] = 1
    action5[6][3] = 1
    state5, delay5, _, _ = env.step(action5)
    print("success")    # debug断点


if __name__ == '__main__':
    # _test_ms_instance()
    # _test_server_node()
    # _test_init_deploy()
    _test_env()