import numpy as np

class EnvConfig:
    """ Alibaba Cluster Trace v2022 — large scale (10 nodes) """

    def __init__(self):
        # id
        self.config_name = "alibaba_largescale"
        # 随机数种子
        self.seed = 1037

        # 时间配置
        self.time_slot_start: int = 0
        self.time_slot_end: int = int(24*60/5) # 5min as a slot

        # 服务器节点配置
        self.node_nums = 10
        self.node_min_cpu_resource = 15
        self.node_max_cpu_resource = 30
        self.node_min_memory_resource = 200
        self.node_max_memory_resource = 300
        self.node2node_min_bandwidth = 3
        self.node2node_max_bandwidth = 3

        # 微服务配置
        self.ms_nums = 10
        self.init_ms_image_list = [3 for _ in range(self.ms_nums)]
        self.max_instance_update_num = 3
        self.ms_min_cpu_resource = 1
        self.ms_max_cpu_resource = 2
        self.ms_min_memory_resource = 10
        self.ms_max_memory_resource = 15
        self.ms_min_lamda = -1
        self.ms_max_lamda = 1
        self.ms_min_mu = 3
        self.ms_max_mu = 5
        self.ms2ms_min_data = 5
        self.ms2ms_max_data = 10

        # 用户、请求链配置
        self.request_flow_nums = 10
        self.min_request_chain_length = 3
        self.max_request_chain_length = 5
        self.estimated_max_lamda = 50
        self.init_lamda = 21
        self.min_request_T = 10
        self.max_request_T = 20
        self.data_path = "data/loads-alibaba-v2022.txt"

        # 开销 (C ~ 2.3 * mean_lambda, 与 twitter/sin 同比例)
        self.cost_w_list = (0.25, 0.1, 1)
        self.C = 50
        self.Q_max = 5
        self.Q_min = 0

        # 模型、算法配置
        self.predicter_window_size = 10
        self.history_lamda_length = min(10, self.node_nums*self.ms_nums)
        self.history_step_length = min(10, self.node_nums*self.ms_nums)
        self.lr = 5e-5
        self.gamma = 0.95
        self.gae_lambda = 0.95
        self.update_epochs = 10
        self.clip_coef = 0.2
        self.clip_vloss = True
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.norm_adv = True

        # 训练配置
        self.device = "cuda:1"
        self.model_path = "model"
        self.is_las = False
        self.num_steps = self.time_slot_end - self.time_slot_start
        self.num_envs = 32
        self.batch_size = int(self.num_envs * self.num_steps)
        self.num_minibatches = 4
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.total_epoches = 20000
        self.total_timesteps = self.total_epoches * self.num_steps * self.num_envs
        self.num_iterations = self.total_timesteps // self.batch_size

        # 奖励配置 (y_weight * 25 / C = 0.2 * 25 / 50 = 0.1)
        self.penalty = -1
        self.y_weight = 0.2
        self.y_weight_train = 0.1
        self.reward_shaping_record_epoch = 100
        self.V = 100
