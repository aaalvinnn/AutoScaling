"""
Ablation study — per-timeslot line charts + mean bar chart (alibaba_largescale)
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'env'))

fontsize = 24
legend_fontsize = 20
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0
markersize = 2.5

ROOT = project_root

from env import environment
from env.configs import config_alibaba_largescale
from methods import PPO_dnn

PPO_dnn.seed_all(1037)

def make_config(**flags):
    c = config_alibaba_largescale.EnvConfig()
    for k, v in flags.items():
        setattr(c, k, v)
    return c

configs = [
    ("Full AutoLFD",  make_config(),                       os.path.join(ROOT, "model/alibaba_largescale/0602/1440/PPO_dnn/model_dnn_best.pth")),
    ("w/o Lyapunov",  make_config(ablation_no_lyapunov_strict=True), os.path.join(ROOT, "model/alibaba_largescale_no_lyapunov_strict/0605/1413/PPO_dnn/model_dnn_best.pth")),
    ("w/o History",   make_config(ablation_no_history=True),  os.path.join(ROOT, "model/alibaba_largescale_no_history/0603/2043/PPO_dnn/model_dnn_best.pth")),
    ("w/o FFD (retrained)",  make_config(ablation_no_ffd=True), os.path.join(ROOT, "model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn/model_dnn_best.pth")),
    ("Full + no-FFD", make_config(ablation_no_ffd=True), os.path.join(ROOT, "model/alibaba_largescale/0602/1440/PPO_dnn/model_dnn_best.pth")),
]

colors = ["#c22f2f", "#376eae", "#449945", "#f8c91c", "#8b5cf6"]
labels_zh = ["AutoLFD (Full)", "w/o Lyapunov", "w/o Historical Data", "w/o FFD (retrained)", "Full + no-FFD"]

all_data = {}

for idx, (name, cfg, model_path) in enumerate(configs):
    environment.CONFIG = cfg
    agent = PPO_dnn.PPOAgent(cfg)
    agent.load(model_path)
    env = environment.DataCenterEnvironment(0, cfg, is_train=False, agent_type="PPO")

    obs, _ = env.reset(seed=1037)
    t_all_list, cost_list, rsr_list = [], [], []
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _, info = env.step(action)
        t_all_list.append(info["t_all"])
        cost_list.append(info["cost"])
        rsr_list.append(info["request_success_rate"])

    all_data[name] = {
        "t_all": np.array(t_all_list),
        "cost": np.array(cost_list),
        "rsr": np.array(rsr_list),
    }
    print(f"{name}: collected {len(t_all_list)} timesteps")

file_names = {
    "Full AutoLFD": "full",
    "w/o Lyapunov": "no_lyapunov",
    "w/o History": "no_history",
    "w/o FFD (retrained)": "no_ffd_retrained",
    "Full + no-FFD": "full_no_ffd",
}
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)
for name, d in all_data.items():
    for metric, arr in d.items():
        np.save(os.path.join(data_dir, f"{file_names[name]}_{metric}.npy"), arr)


def draw_latency():
    plt.figure(figsize=(14, 4))
    for i, (name, cfg, _) in enumerate(configs):
        arr = all_data[name]["t_all"]
        plt.plot(arr, label=labels_zh[i], color=colors[i], linewidth=line_width,
                 zorder=line_zorder, linestyle='--', marker='o', markersize=markersize)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Latency', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='upper right')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    plt.savefig(os.path.join(current_dir, "Latency-ablation-alibaba.pdf"), format="pdf")
    plt.savefig(os.path.join(current_dir, "Latency-ablation-alibaba.png"), format="png", dpi=150)
    print("Saved Latency-ablation-alibaba.pdf")


def draw_cost():
    plt.figure(figsize=(7, 5))
    for i, (name, cfg, _) in enumerate(configs):
        arr = all_data[name]["cost"]
        plt.plot(arr, label=labels_zh[i], color=colors[i], linewidth=line_width, zorder=line_zorder)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    plt.savefig(os.path.join(current_dir, "Cost-ablation-alibaba.pdf"), format="pdf")
    plt.savefig(os.path.join(current_dir, "Cost-ablation-alibaba.png"), format="png", dpi=150)
    print("Saved Cost-ablation-alibaba.pdf")


def draw_rsr():
    plt.figure(figsize=(14, 4))
    for i, (name, cfg, _) in enumerate(configs):
        arr = all_data[name]["rsr"]
        plt.plot(arr, label=labels_zh[i], color=colors[i], linewidth=line_width,
                 zorder=line_zorder, linestyle='--', marker='o', markersize=markersize)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Success Rate', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='lower right')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    plt.savefig(os.path.join(current_dir, "SuccessRate-ablation-alibaba.pdf"), format="pdf")
    plt.savefig(os.path.join(current_dir, "SuccessRate-ablation-alibaba.png"), format="png", dpi=150)
    print("Saved SuccessRate-ablation-alibaba.pdf")


def draw_bar():
    short_labels = ["Full", "no-Lya", "no-Hist", "no-FFD(r)", "Full+noFFD"]
    metrics = [
        ("cost", "Cost", 0),
        ("rsr", "Success Rate", 1),
        ("t_all", "Latency", 2),
    ]
    n_variants = len(configs)
    x = np.arange(n_variants)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for metric_key, metric_label, col in metrics:
        ax = axes[col]
        means = [np.mean(all_data[name][metric_key]) for name, _, _ in configs]
        bars = ax.bar(x, means, 0.6, color=colors, edgecolor='white', linewidth=0.5, zorder=3)

        for bar, val in zip(bars, means):
            if metric_key == "rsr":
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.1%}', ha='center', va='bottom', fontsize=14)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=14)

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=12, rotation=15)
        ax.set_ylabel(metric_label, fontsize=fontsize)
        ax.grid(True, axis='y', zorder=0, alpha=0.5)
        if metric_key == "rsr":
            ax.set_ylim(0, 1.1)

    plt.tight_layout()
    for fmt in ["pdf", "png"]:
        plt.savefig(os.path.join(current_dir, f"Bar-ablation-alibaba.{fmt}"), format=fmt, dpi=150)
    print("Saved Bar-ablation-alibaba.pdf")


if __name__ == '__main__':
    draw_latency()
    draw_cost()
    draw_rsr()
    draw_bar()
