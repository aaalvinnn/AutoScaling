""" Timeslot latency & cost — sin / twitter / alibaba (toggle dataset) """
import matplotlib.pyplot as plt
import numpy as np
import os

fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0
alpha = 1
markersize = 2.5

# dataset = "alibaba"
# dataset = "alibaba"
dataset = "alibaba"

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))

sin_paths = {
    "Large Scale": os.path.join(current_dir, "data/sin_largescale/"),
}
twitter_paths = {
    "Large Scale": os.path.join(current_dir, "data/twitter_largescale/"),
}
alibaba_paths = {
    "Large Scale": os.path.join(current_dir, "data/alibaba_largescale/"),
}

if dataset == "sin":
    data_path = sin_paths
elif dataset == "alibaba":
    data_path = alibaba_paths
else:
    data_path = twitter_paths

scale = "Large Scale"
colors = ["#449945", "#f8c91c", "#376eae", "#8E6EC8", "#c22f2f"]
linestyles = ['--', '--', '--', '--', '--']

def draw_latency(ds="alibaba"):
    d_path = sin_paths if ds == "sin" else (alibaba_paths if ds == "alibaba" else twitter_paths)
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'DeepScaler': None, 'LGDRL': None}
    strs = ['RL Agent', 'Proscale', 'PPA', 'DeepScaler', 'Ours']
    for agent in datas:
        datas[agent] = np.load(os.path.join(d_path[scale], agent, "t_all.npy"))

    plt.figure(figsize=(14, 4))
    for i, agent in enumerate(datas):
        plt.plot(datas[agent], label=strs[i], color=colors[i], linewidth=line_width,
                 zorder=line_zorder, linestyle=linestyles[i], marker='o', markersize=markersize)

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Latency', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper right')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    plt.savefig(os.path.join(current_dir, f"Latency-{ds}-{scale}.pdf"), format="pdf")
    plt.close()

def draw_cost(ds="alibaba"):
    d_path = sin_paths if ds == "sin" else (alibaba_paths if ds == "alibaba" else twitter_paths)
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'DeepScaler': None, 'LGDRL': None}
    strs = ['RL Agent', 'Proscale', 'PPA', 'DeepScaler', 'Ours']
    for agent in datas:
        datas[agent] = np.load(os.path.join(d_path[scale], agent, "cost.npy"))

    plt.figure(figsize=(7, 5))
    all_max = max(np.max(datas[agent]) for agent in datas)
    all_min = min(np.min(datas[agent]) for agent in datas)
    for i, agent in enumerate(datas):
        plt.plot(datas[agent], label=strs[i], color=colors[i], linewidth=line_width, zorder=line_zorder,
                 linestyle=linestyles[i], marker='o', markersize=markersize)

    all_max = max(np.max(d) for d in datas.values())
    all_min = min(np.min(d) for d in datas.values())
    plt.ylim(all_min * 0.85, all_max * 1.45)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, framealpha=0.85, fontsize=lengend_fontsize - 4, loc='upper center', ncol=3, columnspacing=0.8, handletextpad=0.4)
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    plt.savefig(os.path.join(current_dir, f"Cost-{ds}-{scale}.pdf"), format="pdf")
    plt.close()

if __name__ == '__main__':
    for ds in ["sin", "alibaba", "twitter"]:
        draw_latency(ds)
        draw_cost(ds)
        print(f"Saved: {ds}-{scale}")
