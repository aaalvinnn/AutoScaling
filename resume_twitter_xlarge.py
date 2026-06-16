"""twitter_xlargescale 的 warm-start 续训：加载已训练的 model_dnn.pth，从 iter 1 重新跑满 10000。

为什么从 iter 1 而非 3948：
  - checkpoint 只存了模型权重，没存 reward_scaler / optimizer / 迭代计数。
  - 直接从 3948 接着数会让 reward_scaler 处于空状态（mean/var=None），reward_shaping 返回原始 reward，
    与前 3947 epoch 用的归一化 reward 尺度不一致，value function 会失稳。
  - 从 iter 1 重新跑时，前 100 iter 自然重新预热 reward_scaler（record-only），之后恢复正常 shaping；
    而权重已是 3947 epoch 的热状态，相当于“接着练”，省去重学。

用法：
    conda run -n as python resume_twitter_xlarge.py [checkpoint_path]
默认 checkpoint = model/twitter_xlargescale/0616/1251/PPO_dnn/model_dnn.pth（run1 最后 epoch 权重）
"""
import sys
import os

# 双保险：datastruct 在 import 时据此解析 CONFIG
os.environ["AUTOSCALING_CONFIG"] = "twitter_xlargescale"

DEFAULT_CKPT = "model/twitter_xlargescale/0616/1251/PPO_dnn/model_dnn.pth"
checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CKPT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env'))

from env import environment
from env.configs import config_twitter_xlargescale

config = config_twitter_xlargescale.EnvConfig()
environment.CONFIG = config  # 关键：必须在 import PPO_dnn 之前设，PPO_dnn.CONFIG = environment.CONFIG

from methods import PPO_dnn

PPO_dnn.seed_all(config.seed)
agent = PPO_dnn.PPOAgent(config)

print(f"[warm-start] config={config.config_name} node={config.node_nums} ms={config.ms_nums}")
print(f"[warm-start] loading checkpoint: {checkpoint_path}")
agent.load(checkpoint_path)
print("[warm-start] checkpoint loaded ✓, 开始 train()")

PPO_dnn.train(agent)
