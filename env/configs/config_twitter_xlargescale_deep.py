from env.configs.config_twitter_xlargescale import EnvConfig as _TwitterXlargeBase


class EnvConfig(_TwitterXlargeBase):
    """ twitter_xlargescale 的「加深+长训」变体 —— 验证模型容量假设。

    与 twitter_xlargescale 完全相同，仅：
      - config_name 改为 "twitter_xlargescale_deep"（隔离模型目录，不覆盖原 xlarge 结果）
      - total_epoches / num_iterations: 10000 → 20000（更长训练，配合更大模型）
      - reward 用纯 -y（不设 rsr_reward_coef → environment.py 里 getattr 默认 0 → reward=-y；
        之前 _rsr 变体已验证 RSR 项有害，回到原 reward）

    配合 PPO_dnn.py 里 ActorCritic 已从 3 层 MLP 加深到 4 层（容量↑）。
    目的：验证「LGDRL delay/RSR 输给 DeepScaler 是否因模型容量不足」——
    若加宽加深后追上 DeepScaler → 容量是主因；否则是 MLP 架构瓶颈。
    """

    def __init__(self):
        super().__init__()
        self.config_name = "twitter_xlargescale_deep"
        self.total_epoches = 20000
        self.total_timesteps = self.total_epoches * self.num_steps * self.num_envs
        self.num_iterations = self.total_timesteps // self.batch_size
