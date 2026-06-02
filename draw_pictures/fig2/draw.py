import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import os

# 科学绘图包
# plt.style.use(['science','no-latex'])

# 参数设置
fontsize = 24
lengend_fontsize = 21
label_size = 24
bar_width = 0.12
bar_zorder = 3
grid = True
# grid_linewidth = 0.2
grid_zorder = 0

# DATASET = "sin"
DATASET = "twitter"

sin_data_path = {
    "Small Scale": "test_output/sin_smallscale/data/",
    "Middle Scale": "test_output/sin_middlescale/data/",
    "Large Scale": "test_output/sin_largescale/data/",
}

twitter_data_path = {
    "Small Scale": "test_output/twitter_smallscale/data/",
    "Middle Scale": "test_output/twitter_middlescale/data/",
    "Large Scale": "test_output/twitter_largescale/data/"
}

DATA_PATH = sin_data_path if DATASET == "sin" else twitter_data_path

SCALE = "Large Scale"

METRICS = {"Latency": "t_all.npy", "Cost": "cost.npy", "Success Rate": "request_success_rate.npy"}

COLORS = ["#8ECFC9", "#FFBE7A", "#82B0D2", "#FA7F6F"]
# COLORS = ["#8ECFC9", "#f8c91c", "#FA7F6F", "#376eae"]


def get_standard_value(datas: dict, metric: int):
    """ 返回该指标下的最大值用于归一化绘图 """
    buff = []
    for agent in datas.keys():
        buff.append(datas[agent][metric])

    return max(buff)

def draw_fig():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'RL Agent': [], 'Proscale': [], 'HPA': [], 'LGDRL': []}
    strs = ['RL Agent', 'Proscale', 'PPA', 'Ours']
    for agent in datas.keys():
        for metric in METRICS.keys():
            data = np.load(os.path.join(DATA_PATH[SCALE], agent, METRICS[metric]))
            data = np.mean(data)
            print(f"{agent}: {data}")
            datas[agent].append(data)

    # 设置图表的大小
    fig, ax = plt.subplots(figsize=(7, 5))

    # 图 1: Arrival Rate of Requests
    width = bar_width  # 设置条形图宽度
    x = np.arange(len(DATA_PATH.keys()))

    for i, agent in enumerate(datas.keys()):
        values = []
        for j in range(len(METRICS.keys())):
            values.append(datas[agent][j]/get_standard_value(datas, j))

        # ax.bar(x+(i-1.5)*width, datas[agent], width, label=agent, color=COLORS[i], zorder=bar_zorder)
        ax.bar(x+(i-1.5)*width, values, width, label=strs[i], color=COLORS[i], zorder=bar_zorder)

    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    # ax.set_ylabel('Average Request Latency', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(METRICS.keys(), fontsize=fontsize)
    # ax.set_ylim(0, 1,0)  # 延长纵轴上界
    ax.legend(frameon=True, fontsize=lengend_fontsize, loc='lower left')
    plt.grid(grid, zorder=grid_zorder)
    plt.tight_layout()

    # 显示图表
    save_path = os.path.join(current_dir, f"{DATASET}-{SCALE}.pdf")
    plt.savefig(save_path, format="pdf")

if __name__ == '__main__':
    draw_fig()