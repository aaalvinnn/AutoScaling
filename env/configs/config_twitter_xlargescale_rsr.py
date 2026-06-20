from env.configs.config_twitter_xlargescale import EnvConfig as _XlargeBase


class EnvConfig(_XlargeBase):
    """ twitter_xlargescale 的 RSR 增强变体 —— LGDRL reward 加 RSR 项。

    与 twitter_xlargescale 完全相同，仅：
      - config_name 改为 "twitter_xlargescale_rsr"（隔离模型目录，不覆盖原 xlarge 结果）
      - 新增 rsr_reward_coef = 10：environment.py 里 PPO 的 reward 由 -y 改为
        -y + rsr_reward_coef * request_success_rate
    目的：原 LGDRL 只线性压平均 delay，容忍 ~15% 请求因 MS 饱和失败（RSR 0.851）；
    加 RSR 项后强激励去饱和部署，RSR 与 delay 同时改善，对齐/超过 DeepScaler。
    系数 10 为首试值（DeepScaler 用 20；-y≈-19、10*RSR≈8.5，量级可比）。
    """

    def __init__(self):
        super().__init__()
        self.config_name = "twitter_xlargescale_rsr"
        self.rsr_reward_coef = 10
