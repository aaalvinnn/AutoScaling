import sys
import os

ablation_type = sys.argv[1]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'env'))

from env import environment
from env.configs import config_alibaba_largescale

config = config_alibaba_largescale.EnvConfig()

if ablation_type == "no_lyapunov":
    config.ablation_no_lyapunov = True
    config.config_name = "alibaba_largescale_no_lyapunov"
elif ablation_type == "no_lyapunov_strict":
    config.ablation_no_lyapunov_strict = True
    config.config_name = "alibaba_largescale_no_lyapunov_strict"
elif ablation_type == "no_history":
    config.ablation_no_history = True
    config.config_name = "alibaba_largescale_no_history"
elif ablation_type == "no_ffd":
    config.ablation_no_ffd = True
    config.config_name = "alibaba_largescale_no_ffd"
else:
    print(f"Unknown ablation: {ablation_type}")
    sys.exit(1)

config.total_epoches = 10000
config.total_timesteps = config.total_epoches * config.num_steps * config.num_envs
config.num_iterations = config.total_timesteps // config.batch_size
config.device = "cuda:1"

environment.CONFIG = config

from methods import PPO_dnn

PPO_dnn.seed_all(config.seed)
agent = PPO_dnn.PPOAgent(config)
PPO_dnn.train(agent)
