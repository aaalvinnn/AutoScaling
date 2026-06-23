from env.configs.config_alibaba_xlargescale import EnvConfig as _AliXlargeBase


class EnvConfig(_AliXlargeBase):
    """ alibaba_xlargescale 的 RSR 增强变体 —— LGDRL reward 加 RSR 项。

    与 alibaba_xlargescale 完全相同，仅：
      - config_name 改为 "alibaba_xlargescale_rsr"（隔离模型目录）
      - 新增 rsr_reward_coef = 10：environment.py 里 PPO 的 reward 由 -y 改为
        -y + rsr_reward_coef * request_success_rate
    与 config_twitter_xlargescale_rsr / config_sin_xlargescale_rsr 同构，保证三数据集
    大规模实验 reward 一致。
    注意：alibaba trace 基线负载更高 (init_lamda=21)，MS 饱和更重，RSR 结构性上限更低，
    RSR 项收益更关键但也更难补齐——需 eval 后判断系数是否需上调。
    """

    def __init__(self):
        super().__init__()
        self.config_name = "alibaba_xlargescale_rsr"
        self.rsr_reward_coef = 10
