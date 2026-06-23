""" 20节点 twitter baseline 条形图 —— fig2 风格。

读 test_output/twitter_xlargescale_deep/data/{agent}/{t_all,cost,request_success_rate}.npy
（由 draw_pictures/fig4/eval_xlarge_deep.py 生成），按 fig2/draw.py 的样式画归一化分组柱状图，
输出 twitter-XLarge Scale.pdf 到本目录。
"""
import os

import matplotlib.pyplot as plt
import numpy as np

fontsize = 24
lengend_fontsize = 21
label_size = 24
bar_width = 0.12
bar_zorder = 3

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))

DATASET = "twitter"
SCALE = "XLarge Scale"
# twitter 20节点数据目录（fig4 eval 产物）
DATA_PATH = {SCALE: os.path.join(ROOT, "test_output/twitter_xlargescale_deep/data/")}

# 顺序、图例标签、配色 与 fig2/draw.py 的 twitter 分支完全一致
agents_all = ["RL Agent", "Proscale", "HPA", "DeepScaler", "LGDRL"]
labels_all = ["RL Agent", "Proscale", "PPA", "DeepScaler", "Ours"]

METRICS = {"Latency": "t_all.npy", "Cost": "cost.npy", "Success Rate": "request_success_rate.npy"}
COLORS = ["#8ECFC9", "#FFBE7A", "#82B0D2", "#BEB8DC", "#FA7F6F"]


def get_max(datas, idx):
    return max(datas[a][idx] for a in agents_all)


datas = {}
for a in agents_all:
    datas[a] = [np.mean(np.load(os.path.join(DATA_PATH[SCALE], a, m))) for m in METRICS.values()]
    print(f"{a}: Lat={datas[a][0]:.1f} Cost={datas[a][1]:.1f} RSR={datas[a][2]:.3f}")

fig, ax = plt.subplots(figsize=(7, 5))
x = np.arange(len(METRICS))
n = len(agents_all)

for i, a in enumerate(agents_all):
    values = [datas[a][j] / get_max(datas, j) for j in range(len(METRICS))]
    ax.bar(x + (i - (n - 1) / 2) * bar_width, values, bar_width, label=labels_all[i],
           color=COLORS[i], zorder=bar_zorder)

ax.tick_params(axis='x', labelsize=label_size)
ax.tick_params(axis='y', labelsize=label_size)
ax.set_xticks(x)
ax.set_xticklabels(METRICS.keys(), fontsize=fontsize)
ax.legend(frameon=True, fontsize=lengend_fontsize, loc='lower left')
plt.grid(True, zorder=0)
plt.tight_layout()

save_path = os.path.join(current_dir, f"{DATASET}-{SCALE}.pdf")
plt.savefig(save_path, format="pdf")
print(f"Saved: {save_path}")
