""" Baseline bar chart — sin / twitter / alibaba (toggle DATASET) """
import matplotlib.pyplot as plt
import numpy as np
import os

fontsize = 24
lengend_fontsize = 21
label_size = 24
bar_width = 0.12
bar_zorder = 3

# DATASET = "sin"
DATASET = "alibaba"
# DATASET = "alibaba"

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))

sin_paths = {
    "Large Scale": os.path.join(ROOT, "test_output/sin_largescale/data/"),
}
twitter_paths = {
    "Large Scale": os.path.join(ROOT, "test_output/twitter_largescale/data/"),
}
alibaba_paths = {
    "Large Scale": os.path.join(ROOT, "test_output/alibaba_largescale/data/"),
}

if DATASET == "sin":
    DATA_PATH = sin_paths
    agents_all = ["RL Agent", "Proscale", "HPA", "DeepScaler", "LGDRL"]
    labels_all = ["RL Agent", "Proscale", "PPA", "DeepScaler", "Ours"]
elif DATASET == "alibaba":
    DATA_PATH = alibaba_paths
    agents_all = ["HPA", "Proscale", "RL Agent", "DeepScaler", "LGDRL"]
    labels_all = ["PPA", "Proscale", "RL Agent", "DeepScaler", "Ours"]
else:
    DATA_PATH = twitter_paths
    agents_all = ["RL Agent", "Proscale", "HPA", "DeepScaler", "LGDRL"]
    labels_all = ["RL Agent", "Proscale", "PPA", "DeepScaler", "Ours"]

SCALE = "Large Scale"
METRICS = {"Latency": "t_all.npy", "Cost": "cost.npy", "Success Rate": "request_success_rate.npy"}
COLORS  = ["#8ECFC9", "#FFBE7A", "#82B0D2", "#BEB8DC", "#FA7F6F"]

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
    ax.bar(x + (i - (n-1)/2) * bar_width, values, bar_width, label=labels_all[i],
           color=COLORS[i], zorder=bar_zorder)

ax.tick_params(axis='x', labelsize=label_size)
ax.tick_params(axis='y', labelsize=label_size)
ax.set_xticks(x)
ax.set_xticklabels(METRICS.keys(), fontsize=fontsize)
ax.legend(frameon=True, fontsize=lengend_fontsize, loc='lower left')
plt.grid(True, zorder=0)
plt.tight_layout()

save_path = os.path.join(current_dir, f"{DATASET}-Large Scale.pdf")
plt.savefig(save_path, format="pdf")
print(f"Saved: {save_path}")
