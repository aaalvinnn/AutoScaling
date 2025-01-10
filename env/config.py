import numpy as np

class EnvConfig:
    """ 数据中心场景参数设置 """

    def __init__(self):
        # 随机数种子
        self.seed = 1037

        # 时间配置
        self.time_slot_start: int = 0
        self.time_slot_end: int = int(24*60/5) # 5min as a slot

        # 服务器节点配置
        self.node_nums = 10
        self.node_min_cpu_resource = 15   # CPU核数
        self.node_max_cpu_resource = 30
        self.node_min_memory_resource = 200  # 内存 GB
        self.node_max_memory_resource = 300
        self.node2node_min_bandwidth = 1    # 节点间传输带宽
        self.node2node_max_bandwidth = 5

        # 微服务配置
        self.ms_nums = 10
        self.init_ms_image_list = [3 for _ in range(self.ms_nums)]    # 初始的各个微服务实例数量
        self.max_instance_update_num = 2
        self.ms_min_cpu_resource = 1
        self.ms_max_cpu_resource = 2
        self.ms_min_memory_resource = 10
        self.ms_max_memory_resource = 15
        self.ms_min_lamda = -3  # 波动范围
        self.ms_max_lamda = 3
        self.ms_min_mu = 3
        self.ms_max_mu = 5
        self.ms2ms_min_data = 1     # 微服务间依赖数据大小
        self.ms2ms_max_data = 5
        self.weight_cpu_and_memory = 0.1    # FFD对资源排序时，resource = cpu + weight_cpu_and_memory * memory，以平衡数值

        # 用户、请求链配置
        self.request_flow_nums = 5
        self.min_request_chain_length = 3
        self.max_request_chain_length = 8
        self.estimated_max_lamda = 50
        self.init_lamda = 15
        self.min_request_T = 30     # 最大请求时延约束
        self.max_request_T = 50     # 最大请求时延约束
        self.data_path = "data/loads-twitter.txt"

        # 开销
        self.cost_w_list = (0.1, 1, 1)
        self.C = 25     # 服务提供商给出的时间平均长期开销预算

        # 模型、算法配置
        self.predicter_window_size = 5
        self.hitory_lamda_length = min(5, self.node_nums)   # 记录的历史到达率，长度不能大于node_nums，否则无法储存于tensor中；用于模型训练的状态输入
        self.lr = 2e-6
        self.gamma = 0.98
        self.gae_lambda = 0.95
        self.update_epochs = 10
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.norm_adv = True

        # 训练配置
        self.device = "cpu"
        self.model_path = "model"
        self.is_las = False  # 是否展平输出而不采用多输出头
        self.num_steps = self.time_slot_end - self.time_slot_start
        self.num_envs = 8
        self.batch_size = int(self.num_envs * self.num_steps)
        self.num_minibatches = 4
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.total_epoches = 1000
        self.total_timesteps = self.total_epoches * self.num_steps * self.num_envs
        self.num_iterations = self.total_timesteps // self.batch_size

        # 奖励配置
        self.penalty = -50
        self.w_ns_and_delay = 1


