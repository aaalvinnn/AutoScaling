from env.configs.config_alibaba_xlargescale import EnvConfig as _AlibabaXlargeBase


class EnvConfig(_AlibabaXlargeBase):
    """ alibaba_xlargescale 的「加深+长训」变体 —— 验证模型容量假设。

    与 alibaba_xlargescale 完全相同，仅：
      - config_name 改为 "alibaba_xlargescale_deep"（隔离模型目录）
      - total_epoches / num_iterations: 10000 → 20000
      - reward 用纯 -y（不设 rsr_reward_coef → 默认 0 → reward=-y）

    配合 PPO_dnn.py 的 4 层 MLP（容量↑）。与 config_twitter_xlargescale_deep 同构。
    注意：alibaba trace 负载更高 (init_lamda=21)，MS 饱和更重，RSR 结构性上限更低。
    """

    def __init__(self):
        super().__init__()
        self.config_name = "alibaba_xlargescale_deep"
        self.total_epoches = 20000
        self.total_timesteps = self.total_epoches * self.num_steps * self.num_envs
        self.num_iterations = self.total_timesteps // self.batch_size
