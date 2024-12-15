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
        self.node_nums = 5
        self.node_min_cpu_resource = 15   # CPU核数
        self.node_max_cpu_resource = 30
        self.node_min_memory_resource = 200  # 内存 GB
        self.node_max_memory_resource = 300
        self.node2node_min_bandwidth = 3    # 节点间传输带宽
        self.node2node_max_bandwidth = 3

        # 微服务配置
        self.ms_nums = 10
        self.init_ms_image_list = [2 for _ in range(self.ms_nums)]    # 初始的各个微服务实例数量
        self.max_instance_update_num = 2
        self.ms_min_cpu_resource = 1
        self.ms_max_cpu_resource = 2
        self.ms_min_memory_resource = 10
        self.ms_max_memory_resource = 15
        self.ms_min_lamda = 3
        self.ms_max_lamda = 5
        self.ms_min_mu = 3
        self.ms_max_mu = 5
        self.ms2ms_min_data = 0     # 微服务间依赖数据大小
        self.ms2ms_max_data = 3
        self.weight_cpu_and_memory = 0.1    # FFD对资源排序时，resource = cpu + weight_cpu_and_memory * memory，以平衡数值

        # 用户、请求链配置
        self.request_flow_nums = 5
        self.min_request_chain_length = 3
        self.max_request_chain_length = 8

        # 模型配置

        # 奖励配置
        self.penalty = -10
        self.w_ns_and_delay = 0.5


