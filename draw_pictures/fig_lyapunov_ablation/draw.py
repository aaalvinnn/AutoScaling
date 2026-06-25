"""
Lyapunov ablation — per-timeslot line charts (twitter_largescale)
Agents: Full AutoLFD, w/o Lyapunov (strict)
Metrics: Delay (t_all), Cost
"""
import matplotlib.pyplot as plt
import numpy as np
import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'env'))

fig_size = (7, 5)
fontsize = 24
legend_fontsize = 21
label_size = 24
line_width = 1.5
grid_zorder = 0
alpha = 1
markersize = 2.5

ROOT = project_root

from env import environment
from env.configs import config_twitter_largescale
from methods import PPO_dnn

PPO_dnn.seed_all(1037)

# ── configs & model paths ──────────────────────────────────────
def make_config(**flags):
    c = config_twitter_largescale.EnvConfig()
    for k, v in flags.items():
        setattr(c, k, v)
    return c

configs = [
    ("AutoLFD (Full)",     make_config(),                          os.path.join(ROOT, "model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth")),
    ("w/o Lyapunov",       make_config(ablation_no_lyapunov_strict=True), os.path.join(ROOT, "model/twitter_largescale_no_lyapunov_strict/0622/1554/PPO_dnn/model_dnn_3500.pth")),
]

colors     = ["#c22f2f", "#8E6EC8"]
linestyles = ['--',     '--']
labels     = ["AutoLFD (Full)", "w/o Lyapunov"]
ORDER      = ["AutoLFD (Full)", "w/o Lyapunov"]

all_data = {}

# ── rollout for PPO agents ──────────────────────────────────────
for name, cfg, model_path in configs:
    environment.CONFIG = cfg
    agent = PPO_dnn.PPOAgent(cfg)
    agent.load(model_path)
    env = environment.DataCenterEnvironment(0, cfg, is_train=False, agent_type="PPO")

    obs, _ = env.reset(seed=1037)
    t_all_list, cost_list = [], []
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _, info = env.step(action)
        t_all_list.append(info["t_all"])
        cost_list.append(info["cost"])

    all_data[name] = {
        "t_all": np.array(t_all_list),
        "cost": np.array(cost_list),
    }
    cost_arr = np.array(cost_list)
    excess = np.maximum(cost_arr - cfg.C, 0)
    print(f"{name}: collected {len(t_all_list)} timesteps, "
          f"mean_delay={np.mean(t_all_list):.2f}, mean_cost={np.mean(cost_list):.2f}, "
          f"over_budget={np.mean(cost_arr > cfg.C) * 100:.2f}%, excess={np.sum(excess):.2f}")

# ── save raw data ──────────────────────────────────────────────
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)
for name, d in all_data.items():
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    for metric, arr in d.items():
        np.save(os.path.join(data_dir, f"{safe}_{metric}.npy"), arr)

# ── draw per-step delay ────────────────────────────────────────
def draw_latency():
    plt.figure(figsize=fig_size)
    for i, name in enumerate(ORDER):
        arr = all_data[name]["t_all"]
        plt.plot(arr, label=labels[i], color=colors[i], linewidth=line_width,
                 linestyle=linestyles[i], marker='o', markersize=markersize, alpha=alpha)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Delay', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(current_dir, f"Delay-lyapunov-ablation-twitter.{fmt}"),
                    format=fmt, dpi=150 if fmt == "png" else None)
    plt.close()
    print("Saved Delay-lyapunov-ablation-twitter.pdf")


# ── draw per-step cost ─────────────────────────────────────────
def draw_cost():
    plt.figure(figsize=fig_size)
    for i, name in enumerate(ORDER):
        arr = all_data[name]["cost"]
        plt.plot(arr, label=labels[i], color=colors[i], linewidth=line_width,
                 linestyle=linestyles[i], marker='o', markersize=markersize, alpha=alpha)
    plt.axhline(configs[0][1].C, color="black", linestyle="--", linewidth=2.5,
                label=r"Budget $\tilde{C}$=" + f"{configs[0][1].C}")
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(current_dir, f"Cost-lyapunov-ablation-twitter.{fmt}"),
                    format=fmt, dpi=150 if fmt == "png" else None)
    plt.close()
    print("Saved Cost-lyapunov-ablation-twitter.pdf")


if __name__ == '__main__':
    draw_latency()
    draw_cost()
