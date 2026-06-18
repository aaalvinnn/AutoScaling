import sys
import os
import importlib

ablation_type = sys.argv[1]
# 第二个可选参数：数据集/场景名（默认 alibaba_largescale，保持向后兼容）
config_name_arg = sys.argv[2] if len(sys.argv) > 2 else "alibaba_largescale"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))

from env import environment

config_module = importlib.import_module(f"env.configs.config_{config_name_arg}")
config = config_module.EnvConfig()

# (config 上要置位的属性名, config_name 的后缀)
ABLATIONS = {
    "no_lyapunov":        ("ablation_no_lyapunov",        "no_lyapunov"),
    "no_lyapunov_strict": ("ablation_no_lyapunov_strict", "no_lyapunov_strict"),
    "no_history":         ("ablation_no_history",         "no_history"),
    "no_ffd":             ("ablation_no_ffd",             "no_ffd"),
    "oracle_lamda":       ("ablation_oracle_lamda",       "oracle_lamda"),
}
if ablation_type not in ABLATIONS:
    print(f"Unknown ablation: {ablation_type}")
    sys.exit(1)

flag_attr, suffix = ABLATIONS[ablation_type]
setattr(config, flag_attr, True)
config.config_name = f"{config_name_arg}_{suffix}"

config.total_epoches = 10000
config.total_timesteps = config.total_epoches * config.num_steps * config.num_envs
config.num_iterations = config.total_timesteps // config.batch_size
config.device = os.environ.get("AUTOSCALING_DEVICE", "cuda:1")

environment.CONFIG = config

from methods import PPO_dnn

PPO_dnn.seed_all(config.seed)
agent = PPO_dnn.PPOAgent(config)
PPO_dnn.train(agent)
