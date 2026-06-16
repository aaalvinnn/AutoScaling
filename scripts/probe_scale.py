"""规模实验利用率探针：no-op（不扩缩容 ≈ NoScaling 基线）rollout 一整天，
报平均服务强度 rho、拥塞队列数、reward，用于判断扩容后的 config 是否过度供给。

用法（在仓库根目录）：
    AUTOSCALING_CONFIG=twitter_xlargescale conda run -n as python scripts/probe_scale.py
    AUTOSCALING_CONFIG=twitter_largescale   conda run -n as python scripts/probe_scale.py
"""
import os
import sys

# 仓库根加入 sys.path（与训练脚本里的 sys.path hack 同惯例）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 必须在 import env 之前由环境变量选定 config（datastruct 在 import 时解析）
import numpy as np

from env.datastruct import CONFIG
from env.environment import DataCenterEnvironment

cfg = CONFIG
env = DataCenterEnvironment(0, cfg, is_train=True, agent_type="PPO")
# PPO 下 action 为 3-向量 [node, ms, delta_idx]；delta_idx=max_update → delta=0（不扩缩容）
noop = np.array([0, 0, cfg.max_instance_update_num], dtype=np.int64)

asp = env.action_space
asize = asp.n if hasattr(asp, "n") else int(np.prod([s.n for s in asp]))
expected = cfg.node_nums * cfg.ms_nums * (2 * cfg.max_instance_update_num + 1)

print(f"[{cfg.config_name}] node_nums={cfg.node_nums} ms_nums={cfg.ms_nums} "
      f"flows={cfg.request_flow_nums} C={cfg.C} est_max_lamda={cfg.estimated_max_lamda}")
print(f"  action_space size = {asize}  (期望 {expected}, 乘积 {cfg.node_nums}×{cfg.ms_nums}×{2*cfg.max_instance_update_num+1})")
print(f"  obs_space.shape = {env.observation_space.shape}  (期望 (7, {cfg.ms_nums}, {cfg.node_nums}))")

obs, _ = env.reset(seed=cfg.seed)
rhos, congested, rewards = [], [], []
done = False
while not done:
    obs, reward, done, _, info = env.step(noop)
    rhos.append(env._cal_average_service_intensity())
    congested.append(env._get_congested_queue_nums())
    rewards.append(reward)

rhos = np.array(rhos)
print(f"\n--- no-op rollout over {len(rhos)} slots ---")
print(f"  mean rho (平均服务强度)  = {rhos.mean():.3f}   (过大→拥塞, ~0→过度供给)")
print(f"  rho max / min           = {rhos.max():.3f} / {rhos.min():.3f}")
print(f"  拥塞队列数 mean / max   = {np.mean(congested):.2f} / {max(congested)}  (共 {cfg.ms_nums} 个 MS)")
print(f"  reward mean / std       = {np.mean(rewards):.4f} / {np.std(rewards):.4f}")
print(f"  最终 ms_image_list       = {env.ms_image_list}")
