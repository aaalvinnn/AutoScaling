import numpy as np

class EnvConfig:
    """ 数据中心场景参数设置 """

    def __init__(self):
        # 随机数种子
        self.seed = 1037

        # 时间配置
        self.time_slot_start: int = 0
        self.time_slot_end: int = 99

        # 服务器节点配置
        self.node_number = 10
        self.node_min_compute_speed = 1
        self.node_max_compute_speed = 5
        self.node_min_cpu_resource = 10   # CPU核数
        self.node_max_cpu_resource = 20
        self.node_min_memory_resource = 100  # 内存 GB
        self.node_max_memory_resource = 200

        # 微服务配置
        self.ms_number = 5
        self.init_ms_image_list = [2 for _ in range(self.ms_number)]    # 初始的各个微服务实例数量
        self.max_instance_update_num = 2
        self.ms_min_cpu_resource = 1
        self.ms_max_cpu_resource = 2
        self.ms_min_memory_resource = 10
        self.ms_max_memory_resource = 20
        self.weight_cpu_and_memory = 0.1    # FFD对资源排序时，resource = cpu + weight_cpu_and_memory * memory，以平衡数值

        # 用户、请求链配置
        self.request_flow_posion_lambda = 5
        self.min_request_chain_length = 3
        self.max_request_chain_length = 5

        # 模型配置

        # 奖励配置
        self.penalty = -10


