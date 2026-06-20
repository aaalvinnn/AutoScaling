import numpy as np

class EnvConfig:
    """ 数据中心场景参数设置 —— alibaba_xlargescale (20 节点 × 20 微服务)

    由 config_alibaba_largescale (10n×10ms) 全等比放大 2× 得到，与 config_twitter_xlargescale
    同构，用于回应二次意见 ① / R1-1（扁平动作空间 (server, microservice, change_amount) 的大规模可扩展性）。

    全等比缩放（node : ms : flow = 20 : 20 : 20，与 large 的 10:10:10 严格一致）：
      - node_nums         : 10 → 20  (2×)
      - ms_nums           : 10 → 20  (2×，保持 flows/MS = 1.0，per-MS 到达率与 large 完全可比)
      - request_flow_nums : 10 → 20  (2×，= node_nums，保持每节点负载不变)
      - C                 : 35 → 70  (≈ 3.5 × node_nums，与 twitter/sin xlarge 同比例)
      - min/max_request_T : 10/20 → 20/40  (large 的 2 倍，大规模放宽时延约束)
      - estimated_max_lamda : 50 (仅观测归一化用，per-MS 负载与 large 一致故不翻倍)
      - 其余字段（trace loads-alibaba-v2022、init_lamda=21、V=100、奖励权重、训练超参）与 largescale 保持一致。
    """

    def __init__(self):
        # id
        self.config_name = "alibaba_xlargescale"
        # 随机数种子
        self.seed = 1037

        # 时间配置
        self.time_slot_start: int = 0
        self.time_slot_end: int = int(24*60/5) # 5min as a slot

        # 服务器节点配置
        self.node_nums = 20
        self.node_min_cpu_resource = 15   # CPU核数
        self.node_max_cpu_resource = 30
        self.node_min_memory_resource = 200  # 内存 GB
        self.node_max_memory_resource = 300
        self.node2node_min_bandwidth = 3    # 节点间传输带宽 (与 large 一致，公平对比)
        self.node2node_max_bandwidth = 3

        # 微服务配置
        self.ms_nums = 20          # 全等比缩放：node:ms:flow = 20:20:20 (large 为 10:10:10)
        self.init_ms_image_list = [3 for _ in range(self.ms_nums)]    # 初始的各个微服务实例数量
        self.max_instance_update_num = 3    # 与 large 一致
        self.ms_min_cpu_resource = 1
        self.ms_max_cpu_resource = 2
        self.ms_min_memory_resource = 10
        self.ms_max_memory_resource = 15
        self.ms_min_lamda = -1  # 波动范围
        self.ms_max_lamda = 1
        self.ms_min_mu = 3
        self.ms_max_mu = 5
        self.ms2ms_min_data = 5     # 微服务间依赖数据大小
        self.ms2ms_max_data = 10

        # 用户、请求链配置
        self.request_flow_nums = 20
        self.min_request_chain_length = 3
        self.max_request_chain_length = 5
        self.estimated_max_lamda = 50   # 仅用于观测归一化；per-MS 负载与 large 一致，归一化上限同步回归
        self.init_lamda = 21
        self.min_request_T = 20     # 最大请求时延约束 (large 的 2 倍，大规模放宽)
        self.max_request_T = 40     # 最大请求时延约束 (large 的 2 倍，大规模放宽)
        self.data_path = "data/loads-alibaba-v2022.txt"

        # 开销
        self.cost_w_list = (0.25, 0.1, 1)
        self.C = 70     # 服务提供商给出的时间平均长期开销预算 (≈ 3.5 × node_nums)
        self.Q_max = 5
        self.Q_min = 0

        # 模型、算法配置
        self.predicter_window_size = 10
        self.history_lamda_length = min(10, self.node_nums*self.ms_nums)   # 记录的历史到达率，长度不能大于(self.node_nums*self.ms_nums)，否则无法储存于tensor中；用于模型训练的状态输入
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
        self.device = "cuda:0"
        self.model_path = "model"
        self.is_las = False  # 是否展平输出而不采用多输出头
        self.num_steps = self.time_slot_end - self.time_slot_start
        self.num_envs = 8
        self.batch_size = int(self.num_envs * self.num_steps)
        self.num_minibatches = 4
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.total_epoches = 10000
        self.total_timesteps = self.total_epoches * self.num_steps * self.num_envs
        self.num_iterations = self.total_timesteps // self.batch_size

        # 奖励配置 (y_weight * 25 / C = 0.2 * 25 / 70 ≈ 0.071，沿用 largescale 的 0.1)
        self.penalty = -1
        self.y_weight = 0.2
        self.y_weight_train = 0.1
        self.reward_shaping_record_epoch = 100

        # Ablation flags
        self.ablation_no_lyapunov = False
        self.ablation_no_history = False
        self.ablation_no_ffd = False
