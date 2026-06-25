""" 意见 6 —— 部署/路由解耦消融：FFD 时隙曲线。

  - alibaba: 读 fig6 rollout 产物（draw_pictures/fig6/data/*.npy），1×2 子图
  - twitter: 自带 PPO rollout，Cost / Latency 各一张独立图，仅 AutoLFD + w/o FFD 两条线
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'env'))

# ── 样式 ─────────────────────────────────────────────────────────────────────
fontsize = 24
legend_fontsize = 21
label_size = 24
line_width = 1.5
markersize = 2.5
line_zorder = 3
grid_zorder = 0
alpha = 1

N_EARLY = 40   # 早时隙放大图取前多少个时隙

# ── Alibaba 三条线（1×2 子图） ───────────────────────────────────────────────
LINE_DEFS_ALIBABA = [
    ("full",             "AutoLFD (MFFD init)",       "#c22f2f", "--"),
    ("no_ffd_retrained", "w/o FFD (random, retrain)", "#449945", "--"),
    ("full_no_ffd",      "Full + no-FFD (only test)",  "#f8c91c", "--"),
]

# ── Twitter 两条线（Cost/Latency 独立图） ────────────────────────────────────
LINE_DEFS_TWITTER = [
    ("full",             "AutoLFD (MFFD init)",       "#c22f2f", "--"),
    ("no_ffd_retrained", "w/o FFD (random, retrain)", "#449945", "--"),
]


def plot_pair_alibaba(t_all_full, cost_full, n_slots, out_stem, tick_step, xlabel_tag):
    """Alibaba: 1×2（Latency / Cost）子图。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, data_full, ylabel in [
        (axes[0], t_all_full, 'Latency'),
        (axes[1], cost_full,  'Cost'),
    ]:
        for p, lab, col, ls in LINE_DEFS_ALIBABA:
            arr = data_full[p][:n_slots]
            ax.plot(arr, label=lab, color=col, linewidth=line_width, linestyle=ls,
                    marker='o', markersize=markersize, alpha=alpha, zorder=line_zorder)
        ax.set_xticks(np.arange(0, n_slots + 1, tick_step))
        ax.tick_params(axis='both', labelsize=label_size)
        ax.set_xlabel(f"Timeslot ({xlabel_tag})", fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
        ax.grid(True, zorder=grid_zorder)

    plt.tight_layout()
    pdf_path = os.path.join(current_dir, f"{out_stem}.pdf")
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(pdf_path.replace(".pdf", ".png"), format="png", dpi=150)
    plt.close()
    return pdf_path


def plot_single_twitter(data_dict, n_slots, out_stem, tick_step, xlabel_tag, ylabel):
    """Twitter: 单张图（Latency 或 Cost），仅 AutoLFD + w/o FFD 两条线。"""
    fig, ax = plt.subplots(figsize=(7, 5))

    for p, lab, col, ls in LINE_DEFS_TWITTER:
        arr = data_dict[p][:n_slots]
        ax.plot(arr, label=lab, color=col, linewidth=line_width, linestyle=ls,
                marker='o', markersize=markersize, alpha=alpha, zorder=line_zorder)

    ax.set_xticks(np.arange(0, n_slots + 1, tick_step))
    ax.tick_params(axis='both', labelsize=label_size)
    ax.set_xlabel(f"Timeslot ({xlabel_tag})", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    ax.grid(True, zorder=grid_zorder)

    plt.tight_layout()
    pdf_path = os.path.join(current_dir, f"{out_stem}.pdf")
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(pdf_path.replace(".pdf", ".png"), format="png", dpi=150)
    plt.close()
    return pdf_path


# ══════════════════════════════════════════════════════════════════════════════
# 1) Alibaba —— 从 fig6/data 读取现成 npy
# ══════════════════════════════════════════════════════════════════════════════
DATA_DIR_ALIBABA = os.path.join(ROOT, "draw_pictures", "fig6", "data")

t_all_alibaba = {p: np.load(os.path.join(DATA_DIR_ALIBABA, f"{p}_t_all.npy")) for p, _, _, _ in LINE_DEFS_ALIBABA}
cost_alibaba  = {p: np.load(os.path.join(DATA_DIR_ALIBABA, f"{p}_cost.npy"))  for p, _, _, _ in LINE_DEFS_ALIBABA}
TOTAL_A = int(min(min(a.shape[0] for a in t_all_alibaba.values()),
                  min(a.shape[0] for a in cost_alibaba.values())))

print("=== Alibaba ===")
p_early_a = plot_pair_alibaba(t_all_alibaba, cost_alibaba, N_EARLY,
                               "EarlySlots-FFD-ablation-alibaba", tick_step=10, xlabel_tag="early")
p_full_a  = plot_pair_alibaba(t_all_alibaba, cost_alibaba, TOTAL_A,
                               "Full-FFD-ablation-alibaba",       tick_step=40, xlabel_tag="full")
print(f"  早时隙 → {p_early_a}  (前 {N_EARLY} 时隙)")
print(f"  完整时隙 → {p_full_a}  (共 {TOTAL_A} 时隙)")

for p, lab, _, _ in LINE_DEFS_ALIBABA:
    print(f"  {lab:30s} Lat: 前10={t_all_alibaba[p][:10].mean():6.2f}  全程={t_all_alibaba[p].mean():6.2f}   "
          f"Cost: 前10={cost_alibaba[p][:10].mean():6.2f}  全程={cost_alibaba[p].mean():6.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 2) Twitter —— 自带 PPO rollout，Cost / Latency 各一张独立图，仅两条线
# ══════════════════════════════════════════════════════════════════════════════
from env import environment
from env.configs import config_twitter_largescale
from methods import PPO_dnn

PPO_dnn.seed_all(1037)


def make_config(**flags):
    c = config_twitter_largescale.EnvConfig()
    for k, v in flags.items():
        setattr(c, k, v)
    return c


twitter_configs = [
    ("full",             make_config(),
     os.path.join(ROOT, "model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth")),
    ("no_ffd_retrained", make_config(ablation_no_ffd=True),
     os.path.join(ROOT, "model/twitter_largescale_no_ffd/0622/1143/PPO_dnn/model_dnn_best.pth")),
]

t_all_tw = {}
cost_tw = {}

print("\n=== Twitter ===")
for p, cfg, model_path in twitter_configs:
    environment.CONFIG = cfg
    agent = PPO_dnn.PPOAgent(cfg)
    agent.load(model_path)
    env = environment.DataCenterEnvironment(0, cfg, is_train=False, agent_type="PPO")

    obs, _ = env.reset(seed=1037)
    t_list, c_list = [], []
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _, info = env.step(action)
        t_list.append(info["t_all"])
        c_list.append(info["cost"])

    t_all_tw[p] = np.array(t_list)
    cost_tw[p] = np.array(c_list)
    print(f"  {p}: {len(t_list)} steps, lat_mean={np.mean(t_list):.2f}, cost_mean={np.mean(c_list):.2f}")

TOTAL_T = min(len(a) for a in t_all_tw.values())

# EarlySlots — Latency / Cost 各一张独立图
p_lat_e = plot_single_twitter(t_all_tw, N_EARLY,
                              "EarlySlots-Latency-FFD-ablation-twitter", tick_step=10, xlabel_tag="early",
                              ylabel="Latency")
p_cost_e = plot_single_twitter(cost_tw, N_EARLY,
                               "EarlySlots-Cost-FFD-ablation-twitter", tick_step=10, xlabel_tag="early",
                               ylabel="Cost")

# Full — Latency / Cost 各一张独立图
p_lat_f = plot_single_twitter(t_all_tw, TOTAL_T,
                              "Full-Latency-FFD-ablation-twitter", tick_step=40, xlabel_tag="full",
                              ylabel="Latency")
p_cost_f = plot_single_twitter(cost_tw, TOTAL_T,
                               "Full-Cost-FFD-ablation-twitter", tick_step=40, xlabel_tag="full",
                               ylabel="Cost")

print(f"  EarlySlots Latency → {p_lat_e}")
print(f"  EarlySlots Cost    → {p_cost_e}")
print(f"  Full Latency        → {p_lat_f}")
print(f"  Full Cost           → {p_cost_f}")
