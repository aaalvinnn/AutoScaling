from env.configs.config_sin_xlargescale import EnvConfig as _SinXlargeBase


class EnvConfig(_SinXlargeBase):
    """ sin_xlargescale 的 RSR 增强变体 —— LGDRL reward 加 RSR 项。

    与 sin_xlargescale 完全相同，仅：
      - config_name 改为 "sin_xlargescale_rsr"（隔离模型目录）
      - 新增 rsr_reward_coef = 10：environment.py 里 PPO 的 reward 由 -y 改为
        -y + rsr_reward_coef * request_success_rate
    与 config_twitter_xlargescale_rsr 同构，保证三数据集大规模实验 reward 一致。
    系数 10 为首试值（DeepScaler 用 20；-y≈-16、10*RSR≈9，量级可比）。
    """

    def __init__(self):
        super().__init__()
        self.config_name = "sin_xlargescale_rsr"
        self.rsr_reward_coef = 10
