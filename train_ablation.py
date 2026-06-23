import sys
import os
import importlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))

from env import environment
from methods.train_checkpoint import find_latest_resume_dir

# (config 上要置位的属性名, config_name 的后缀)
ABLATIONS = {
    "no_lyapunov":        ("ablation_no_lyapunov",        "no_lyapunov"),
    "no_lyapunov_strict": ("ablation_no_lyapunov_strict", "no_lyapunov_strict"),
    "no_history":         ("ablation_no_history",         "no_history"),
    "no_ffd":             ("ablation_no_ffd",             "no_ffd"),
    "oracle_lamda":       ("ablation_oracle_lamda",       "oracle_lamda"),
}


def build_ablation_config(ablation_type, config_name_arg="alibaba_largescale"):
    """构造指定消融变体的 EnvConfig（置 flag、改 config_name、锁定 epoches/device）。"""
    if ablation_type not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {ablation_type}; expected one of {sorted(ABLATIONS)}")
    config_module = importlib.import_module(f"env.configs.config_{config_name_arg}")
    config = config_module.EnvConfig()
    flag_attr, suffix = ABLATIONS[ablation_type]
    setattr(config, flag_attr, True)
    config.config_name = f"{config_name_arg}_{suffix}"

    config.total_epoches = 10000
    config.total_timesteps = config.total_epoches * config.num_steps * config.num_envs
    config.num_iterations = config.total_timesteps // config.batch_size
    config.device = os.environ.get("AUTOSCALING_DEVICE", "cuda:1")
    config.checkpoint_interval = int(os.environ.get("AUTOSCALING_CHECKPOINT_INTERVAL", "5000"))
    return config


# 注意：执行逻辑必须包在 if __name__ == "__main__": 里。
# 训练用 spawn 后端的 AsyncVectorEnv，其 worker 会 re-import 本模块（__mp_main__），
# 若此处不 guard，worker 会重新执行 train() → 递归 spawn → ConnectionResetError。
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python train_ablation.py <ablation_type> [config_name] [--auto-resume]")
        print(f"  ablation_type ∈ {sorted(ABLATIONS)}; config_name 默认 alibaba_largescale")
        sys.exit(1)
    ablation_type = sys.argv[1]
    # 第二个可选参数：数据集/场景名（默认 alibaba_largescale，保持向后兼容）
    config_name_arg = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "alibaba_largescale"

    config = build_ablation_config(ablation_type, config_name_arg)
    environment.CONFIG = config

    from methods import PPO_dnn

    PPO_dnn.seed_all(config.seed)
    agent = PPO_dnn.PPOAgent(config)
    # --auto-resume: 从最近一次 train_state.pt 续跑（供外层重启循环调用，同 PPO_dnn.py）
    resume_dir = find_latest_resume_dir(config, "PPO_dnn") if "--auto-resume" in sys.argv else None
    PPO_dnn.train(agent, resume_dir=resume_dir)
