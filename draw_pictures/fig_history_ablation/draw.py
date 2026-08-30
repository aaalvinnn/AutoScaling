"""
Prediction / history ablation (#3) — per-timeslot line charts (twitter_largescale)
Agents: AutoLFD (Full), Oracle (known current), w/o History (blind)
Metrics: Delay (t_all), Cost
"""
import matplotlib.pyplot as plt
import numpy as np
import os, sys
from matplotlib.ticker import MultipleLocator

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'env'))

fig_size = (7, 5)
fontsize = 24
legend_fontsize = 18
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

def make_config(**flags):
    c = config_twitter_largescale.EnvConfig()
    for k, v in flags.items():
        setattr(c, k, v)
    return c

configs = [
    ("AutoLFD (Full)",  make_config(),                          os.path.join(ROOT, "model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth")),
    ("Oracle",          make_config(ablation_oracle_lamda=True),os.path.join(ROOT, "model/twitter_largescale_oracle_lamda/0617/1206/PPO_dnn/model_dnn_best.pth")),
    ("w/o History",     make_config(ablation_no_history=True),  os.path.join(ROOT, "model/twitter_largescale_no_history/0617/1206/PPO_dnn/model_dnn_best.pth")),
]

colors     = ["#c22f2f", "#449945", "#376eae"]
linestyles = ['--',      '--',      '--']
labels     = ["AutoLFD (Full)", "Oracle (known current)", "w/o History"]
ORDER      = ["AutoLFD (Full)", "Oracle", "w/o History"]

all_data = {}

# ── PPO agents ──
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
    all_data[name] = {"t_all": np.array(t_all_list), "cost": np.array(cost_list)}
    print(f"{name}: {len(t_all_list)} steps, mean_delay={np.mean(t_all_list):.2f}, mean_cost={np.mean(cost_list):.2f}")

# ── save raw ──
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)
for name, d in all_data.items():
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    for metric, arr in d.items():
        np.save(os.path.join(data_dir, f"{safe}_{metric}.npy"), arr)

def draw_latency():
    fig, ax = plt.subplots(figsize=fig_size)
    all_max = max(np.max(all_data[name]["t_all"]) for name in ORDER)
    all_min = min(np.min(all_data[name]["t_all"]) for name in ORDER)
    for i, name in enumerate(ORDER):
        arr = all_data[name]["t_all"]
        ax.plot(arr, label=labels[i], color=colors[i], linewidth=line_width,
                linestyle=linestyles[i], marker='o', markersize=markersize, alpha=alpha)
    ax.set_ylim(all_min * 0.85, all_max * 1.35)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=label_size)
    ax.set_ylabel('Latency', fontsize=fontsize); ax.set_xlabel("Timeslot", fontsize=fontsize)
    ax.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    ax.grid(True, zorder=grid_zorder)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(current_dir, f"Delay-history-ablation-twitter.{fmt}"),
                    format=fmt, dpi=150 if fmt == "png" else None, bbox_inches='tight')
    plt.close()
    print("Saved Delay-history-ablation-twitter.pdf")

def draw_cost():
    fig, ax = plt.subplots(figsize=fig_size)
    all_max = max(np.max(all_data[name]["cost"]) for name in ORDER)
    all_min = min(np.min(all_data[name]["cost"]) for name in ORDER)
    for i, name in enumerate(ORDER):
        arr = all_data[name]["cost"]
        ax.plot(arr, label=labels[i], color=colors[i], linewidth=line_width,
                linestyle=linestyles[i], marker='o', markersize=markersize, alpha=alpha)
    ax.set_ylim(all_min * 0.85, all_max * 1.3)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.tick_params(axis='both', labelsize=label_size)
    ax.set_ylabel('Cost', fontsize=fontsize); ax.set_xlabel("Timeslot", fontsize=fontsize)
    ax.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    ax.grid(True, zorder=grid_zorder)
    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(current_dir, f"Cost-history-ablation-twitter.{fmt}"),
                    format=fmt, dpi=150 if fmt == "png" else None, bbox_inches='tight')
    plt.close()
    print("Saved Cost-history-ablation-twitter.pdf")

if __name__ == '__main__':
    draw_latency()
    draw_cost()
