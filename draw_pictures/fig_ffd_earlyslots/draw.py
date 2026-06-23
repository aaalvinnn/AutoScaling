""" 意见 6 —— 部署/路由解耦消融：FFD 三条线的时隙曲线（alibaba_largescale）。

读 fig6 rollout 产物（draw_pictures/fig6/data/*.npy），画 Full / w/o FFD(retrain) / Full+no-FFD(only-test)
三条线的 Latency(t_all) 与 Cost：
  - EarlySlots 图：前 N_EARLY 个时隙放大（部署质量对早期推理的影响 —— 师兄预测 random 前几时隙更差）
  - Full 图：完整时隙曲线（看全貌 / 是否恢复）

师兄意见 6：随机初始部署在前几个时隙应明显差于 MFFD；早时隙图单独出，这里顺便补完整时隙对比。
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# ── 样式（与 fig6 一致）──────────────────────────────────────────────────────
fontsize = 24
legend_fontsize = 18
label_size = 22
line_width = 1.8
markersize = 3
line_zorder = 3
grid_zorder = 0
alpha = 1

N_EARLY = 40   # 早时隙放大图取前多少个时隙

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))
DATA_DIR = os.path.join(ROOT, "draw_pictures", "fig6", "data")

# 三条线：键 → (图例, fig6/data 文件前缀, 颜色, 线型)
lines = [
    ("full",             "AutoLFD (MFFD init)",       "#c22f2f", "-"),   # 红 实线
    ("no_ffd_retrained", "w/o FFD (random, retrain)", "#449945", "--"),  # 绿 虚线
    ("full_no_ffd",      "Full + no-FFD (only test)",  "#f8c91c", "--"),  # 黄 虚线
]

# 读取完整时隙数组（不截断）
t_all_full = {p: np.load(os.path.join(DATA_DIR, f"{p}_t_all.npy")) for p, _, _, _ in lines}
cost_full  = {p: np.load(os.path.join(DATA_DIR, f"{p}_cost.npy"))  for p, _, _, _ in lines}
TOTAL = int(min(min(a.shape[0] for a in t_all_full.values()),
                min(a.shape[0] for a in cost_full.values())))


def plot_pair(n_slots, out_stem, tick_step, xlabel_tag):
    """画一张 1×2（Latency / Cost）图，取前 n_slots 个时隙。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, data_full, ylabel in [
        (axes[0], t_all_full, 'Latency'),
        (axes[1], cost_full,  'Cost'),
    ]:
        for p, lab, col, ls in lines:
            arr = data_full[p][:n_slots]
            ax.plot(arr, label=lab, color=col, linewidth=line_width, linestyle=ls,
                    marker='o', markersize=markersize, alpha=alpha, zorder=line_zorder)
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


p_early = plot_pair(N_EARLY, "EarlySlots-FFD-ablation-alibaba", tick_step=10, xlabel_tag="early")
p_full  = plot_pair(TOTAL,   "Full-FFD-ablation-alibaba",       tick_step=40, xlabel_tag="full")

print(f"早时隙图 → {p_early}  (前 {N_EARLY} 时隙)")
print(f"完整时隙 → {p_full}  (共 {TOTAL} 时隙)")
print(f"\n汇总（首 10 时隙均值 vs 全程均值）：")
for p, lab, _, _ in lines:
    print(f"  {lab:30s} Lat: 前10={t_all_full[p][:10].mean():6.2f}  全程={t_all_full[p].mean():6.2f}   "
          f"Cost: 前10={cost_full[p][:10].mean():6.2f}  全程={cost_full[p].mean():6.2f}")
