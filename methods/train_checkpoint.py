"""训练全状态 checkpoint / 续跑工具（PPO / DeepScaler / SAC 共用）。

把「权重 + 优化器 + reward shaping + 计数器 + RNG（+ 目标网/alpha）」整体原子落盘为
`train_state.pt`，供 AsyncVectorEnv 崩溃后被外层重启循环精确续跑。

设计要点：
- 原子写（torch.save 临时文件 + os.replace）：崩溃可能恰好发生在 save 中途，
  os.replace 保证 train_state.pt 要么是上一份完整态、要么是新的完整态，绝不半写。
- weights_only=False 仅用于本模块自写的 train_state.pt（payload 含 python dict / numpy
  RNG 元组，自写自读可信）；评估用的 per-module .pth 仍各自 weights_only=True。
"""

import glob
import os
import random

import numpy as np
import torch

TRAIN_STATE_NAME = "train_state.pt"


def save_train_state(path, payload):
    """原子地保存完整训练状态字典到 path。

    Args:
        path:   目标文件路径（通常 {save_path}/train_state.pt）。
        payload: dict，至少含 model/optimizer/reward_scaler/iteration/global_step/rng。
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    # CPU 上序列化再落盘，避免大 GPU tensor 直接 pickle 的额外开销/风险
    torch.save(payload, tmp)
    os.replace(tmp, path)  # 原子替换：要么旧要么新，绝不半写


def load_train_state(path, map_location=None):
    """读取 train_state.pt（weights_only=False，仅用于自写文件）。"""
    return torch.load(path, map_location=map_location, weights_only=False)


# ── RNG 快照 / 恢复 ───────────────────────────────────────────────────────────

def snapshot_rng():
    """捕获 python / numpy / torch / cuda RNG 状态。"""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    """恢复 RNG 状态；state 为 None 则跳过。

    - python / numpy / torch-CPU 状态可精确恢复。
    - torch-CUDA 主进程 RNG：torch 2.11 下 set_rng_state_all 要求 legacy
      cuda.ByteTensor 类型，而 pickle round-trip 后无法重建该类型，故此处 best-effort，
      失败则跳过（影响极小：仅主进程采样噪声，env worker 自带 seed，每轮自校正）。
    """
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_ts = state["torch"]
    if hasattr(cpu_ts, "cpu"):
        cpu_ts = cpu_ts.cpu().byte()
    torch.set_rng_state(cpu_ts)
    cuda_states = state.get("torch_cuda")
    if cuda_states and torch.cuda.is_available():
        try:
            cuda_states = [x.cuda().byte() if hasattr(x, "cuda") else x for x in cuda_states]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception as e:
            print(f"[resume] 跳过 CUDA RNG 恢复（{type(e).__name__}: {e}）", flush=True)


# ── RewardScaler 序列化 ──────────────────────────────────────────────────────
# 三个算法的 RewardScaler 都含 mean/var/count；buffer 不存（record 阶段会重新采集）。

def reward_scaler_to_dict(rs):
    return {"mean": rs.mean, "var": rs.var, "count": getattr(rs, "count", 0)}


def load_reward_scaler(rs, d):
    """把字典里的 mean/var/count 写回 RewardScaler 对象。"""
    rs.mean = d.get("mean", rs.mean)
    rs.var = d.get("var", rs.var)
    if hasattr(rs, "count"):
        rs.count = d.get("count", 0)


# ── 续跑目录定位 ──────────────────────────────────────────────────────────────

def find_latest_resume_dir(config, algo_subdir):
    """在 {model_path}/{config_name}/*/*/{algo_subdir}/ 下找最新 train_state.pt 所在目录。

    Args:
        config:        EnvConfig（用 config.model_path / config.config_name）。
        algo_subdir:   "PPO_dnn" | "DeepScaler" | "SAC"（与各 train() save_path 末段一致）。
    Returns:
        最新 train_state.pt 所在目录（str），没有则 None。
    """
    pattern = os.path.join(
        config.model_path,
        config.config_name,
        "*",
        "*",
        algo_subdir,
        TRAIN_STATE_NAME,
    )
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    ckpts.sort(key=os.path.getmtime)
    return os.path.dirname(ckpts[-1])
