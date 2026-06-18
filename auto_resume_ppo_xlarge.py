"""PPO 自动 warm-start 续跑：自动查找最新 model_dnn.pth 加载后训练。

设计为被 bash 循环反复调用：崩了 → 外层重启 → 本脚本找最新 checkpoint 续跑。
reward_scaler 每次从头预热(前100 iter record-only)，权重是热的，相当于接着练。

用法：python auto_resume_ppo_xlarge.py
"""
import sys
import os
import glob

os.environ["AUTOSCALING_CONFIG"] = "twitter_xlargescale"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'env'))

from env import environment
from env.configs import config_twitter_xlargescale
from methods import PPO_dnn

config = config_twitter_xlargescale.EnvConfig()
environment.CONFIG = config

# 自动查找最新的 model_dnn.pth（跨所有 0618/HHMM 目录）
ckpts = sorted(
    glob.glob("model/twitter_xlargescale/*/??？？/PPO_dnn/model_dnn.pth"),
    key=os.path.getmtime,
)
# 同时考虑 model_dnn_best.pth 作为 fallback
all_ckpts = sorted(
    glob.glob("model/twitter_xlargescale/*/????/PPO_dnn/model_dnn*.pth"),
    key=os.path.getmtime,
)

PPO_dnn.seed_all(config.seed)
agent = PPO_dnn.PPOAgent(config)

if all_ckpts:
    ckpt = all_ckpts[-1]  # 最新修改的
    print(f"[auto-resume] loading latest checkpoint: {ckpt}")
    try:
        agent.load(ckpt)
        print("[auto-resume] loaded ✓")
    except Exception as e:
        print(f"[auto-resume] load failed ({e}), cold start")
else:
    print("[auto-resume] no checkpoint found, cold start")

PPO_dnn.train(agent)
