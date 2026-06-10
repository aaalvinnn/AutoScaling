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
    "Large Scale": os.path.join(ROOT, "test_output/sin_largescale/data/"),
}
twitter_paths = {
    "Large Scale": os.path.join(ROOT, "test_output/twitter_largescale/data/"),
}
alibaba_paths = {
    "Large Scale": os.path.join(ROOT, "test_output/alibaba_largescale/data/"),
}

if dataset == "sin":
    data_path = sin_paths
elif dataset == "alibaba":
    data_path = alibaba_paths
else:
    data_path = twitter_paths

scale = "Large Scale"
colors = ["#449945", "#f8c91c", "#376eae", "#c22f2f", "#8E6EC8"]
linestyles = ['--', '--', '--', '--', '-.']

def draw_latency():
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'DeepScaler': None, 'LGDRL': None}
    strs = ['RL Agent', 'Proscale', 'PPA', 'DeepScaler', 'Ours']
    for agent in datas:
        datas[agent] = np.load(os.path.join(data_path[scale], agent, "t_all.npy"))

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
    plt.savefig(os.path.join(current_dir, f"Latency-{dataset}-{scale}.pdf"), format="pdf")

def draw_cost():
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'DeepScaler': None, 'LGDRL': None}
    strs = ['RL Agent', 'Proscale', 'PPA', 'DeepScaler', 'Ours']
    for agent in datas:
        datas[agent] = np.load(os.path.join(data_path[scale], agent, "cost.npy"))

    plt.figure(figsize=(7, 5))
    for i, agent in enumerate(datas):
        plt.plot(datas[agent], label=strs[i], color=colors[i], linewidth=line_width, zorder=line_zorder,
                 linestyle=linestyles[i])

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    plt.savefig(os.path.join(current_dir, f"Cost-{dataset}-{scale}.pdf"), format="pdf")

if __name__ == '__main__':
    draw_latency()
    draw_cost()
    print(f"Saved: {dataset}-{scale}")
