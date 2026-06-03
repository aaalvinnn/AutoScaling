""" Load curves: Twitter + Sin + Alibaba """
import matplotlib.pyplot as plt
import numpy as np
import os

fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 2.5
line_zorder = 3
grid_zorder = 0

def read_data(path):
    return np.loadtxt(path).tolist()

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))

datasets = [
    ("Twitter", os.path.join(ROOT, "data/loads-twitter.txt"), "#82B0D2"),
    ("Sin",     os.path.join(ROOT, "data/loads-sin.txt"),     "#8ECFC9"),
    ("Alibaba", os.path.join(ROOT, "data/loads-alibaba-v2022.txt"), "#FA7F6F"),
]

fig, ax = plt.subplots(figsize=(14, 4))
for label, path, color in datasets:
    data = read_data(path)
    ax.plot(data, label=label, linewidth=line_width, color=color, zorder=line_zorder)

ax.tick_params(axis='x', labelsize=label_size)
ax.tick_params(axis='y', labelsize=label_size)
ax.set_xlabel("Timeslot", fontsize=fontsize)
ax.set_ylabel("Lambda", fontsize=fontsize)
ax.grid(True, zorder=grid_zorder)

plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
plt.tight_layout()

save_path = os.path.join(current_dir, "loads-alibaba.pdf")
plt.savefig(save_path, format="pdf")
print(f"Saved: {save_path}")
