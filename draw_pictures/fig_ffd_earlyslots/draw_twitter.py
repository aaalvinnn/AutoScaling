""" 意见 6 —— twitter_largescale 版：FFD 三条线的时隙曲线（与 alibaba 版对齐）。

加载 twitter 上同条件（y_weight_train=0.1）训出的 Full 与 w/o FFD 模型，跑 rollout 收集逐时隙
Latency(t_all)/Cost，画：
  - EarlySlots 图：前 N_EARLY 个时隙放大（random 前几时隙更差）
  - Full 图：完整时隙曲线（看全貌）

三条线：Full(MFFD init) / w/o FFD(random, retrain) / Full+no-FFD(only test)。
数据自跑自存到 data/（alibaba 版读的是 fig6/data，twitter 没有现成 rollout，故本脚本自跑）。
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'env'))

from env import environment
from env.configs import config_twitter_largescale
from methods import PPO_dnn

PPO_dnn.seed_all(1037)

ROOT = project_root

# ── 样式（与 fig5/delta.py 的单图画幅一致）───────────────────────────────────
fig_size = (7, 5)
fontsize = 15
legend_fontsize = 15
label_size = 12
line_width = 1.5
markersize = 2
line_zorder = 3
grid_zorder = 0
alpha = 1
N_EARLY = 40

# 三条线：(图例, flags, 模型路径, 颜色, 线型)
configs = [
    ("AutoLFD (MFFD init)",        {},
        os.path.join(ROOT, "model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth"),
        "#c22f2f", "-"),
    ("w/o FFD (random, retrain)",  {"ablation_no_ffd": True},
        os.path.join(ROOT, "model/twitter_largescale_no_ffd/0622/1143/PPO_dnn/model_dnn_best.pth"),
        "#449945", "--"),
]


def make_config(**flags):
    c = config_twitter_largescale.EnvConfig()
    for k, v in flags.items():
        setattr(c, k, v)
    return c


# ── rollout ───────────────────────────────────────────────────────────────────
file_keys = {"AutoLFD (MFFD init)": "full",
             "w/o FFD (random, retrain)": "no_ffd_retrained",
             "Full + no-FFD (only test)": "full_no_ffd"}
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)

t_all, cost = {}, {}
for name, flags, model_path, _, _ in configs:
    cfg = make_config(**flags)
    environment.CONFIG = cfg
    agent = PPO_dnn.PPOAgent(cfg)
    agent.load(model_path)
    env = environment.DataCenterEnvironment(0, cfg, is_train=False, agent_type="PPO")

    obs, _ = env.reset(seed=1037)
    tl, cl = [], []
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _, info = env.step(action)
        tl.append(info["t_all"])
        cl.append(info["cost"])
    t_all[name] = np.array(tl)
    cost[name] = np.array(cl)
    # 存 npy（键名与 alibaba 版 fig6/data 对齐）
    k = file_keys[name]
    np.save(os.path.join(data_dir, f"{k}_t_all.npy"), t_all[name])
    np.save(os.path.join(data_dir, f"{k}_cost.npy"), cost[name])
    print(f"{name}: {len(tl)} steps, lat_mean={np.mean(tl):.2f}, cost_mean={np.mean(cl):.2f}")

TOTAL = int(min(min(a.shape[0] for a in t_all.values()),
                min(a.shape[0] for a in cost.values())))
N_EARLY = min(N_EARLY, TOTAL)


def plot_metric(data, n_slots, out_stem, tick_step, xlabel_tag, ylabel):
    fig, ax = plt.subplots(figsize=fig_size)
    for name, _, _, col, ls in configs:
        ax.plot(data[name][:n_slots], label=name, color=col, linewidth=line_width,
                linestyle=ls, marker='o', markersize=markersize, alpha=alpha, zorder=line_zorder)
    ax.set_xticks(np.arange(0, n_slots + 1, tick_step))
    ax.tick_params(axis='both', labelsize=label_size)
    ax.set_xlabel(f"Timeslot ({xlabel_tag})", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(frameon=True, fontsize=legend_fontsize, loc='best')
    ax.grid(True, zorder=grid_zorder)
    plt.tight_layout()
    pdf_path = os.path.join(current_dir, f"{out_stem}.pdf")
    plt.savefig(pdf_path, format="pdf")
    plt.savefig(pdf_path.replace(".pdf", ".png"), format="png", dpi=150)
    plt.close()
    return pdf_path


p_early_latency = plot_metric(t_all, N_EARLY, "EarlySlots-Latency-FFD-ablation-twitter",
                              tick_step=10, xlabel_tag="early", ylabel="Latency")
p_early_cost = plot_metric(cost, N_EARLY, "EarlySlots-Cost-FFD-ablation-twitter",
                           tick_step=10, xlabel_tag="early", ylabel="Cost")
p_full_latency = plot_metric(t_all, TOTAL, "Full-Latency-FFD-ablation-twitter",
                             tick_step=max(20, TOTAL // 8), xlabel_tag="full", ylabel="Latency")
p_full_cost = plot_metric(cost, TOTAL, "Full-Cost-FFD-ablation-twitter",
                          tick_step=max(20, TOTAL // 8), xlabel_tag="full", ylabel="Cost")

print(f"\n早时隙延迟图 → {p_early_latency}  (前 {N_EARLY} 时隙)")
print(f"早时隙成本图 → {p_early_cost}  (前 {N_EARLY} 时隙)")
print(f"完整时隙延迟图 → {p_full_latency}  (共 {TOTAL} 时隙)")
print(f"完整时隙成本图 → {p_full_cost}  (共 {TOTAL} 时隙)")
print(f"\n汇总（首 10 时隙均值 vs 全程均值）：")
for name, _, _, _, _ in configs:
    print(f"  {name:30s} Lat: 前10={t_all[name][:10].mean():6.2f}  全程={t_all[name].mean():6.2f}   "
          f"Cost: 前10={cost[name][:10].mean():6.2f}  全程={cost[name].mean():6.2f}")
