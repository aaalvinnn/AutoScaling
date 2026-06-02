import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import os

# 科学绘图包
# plt.style.use(['science','no-latex'])

# 参数设置
fontsize = 20
lengend_fontsize = 15
label_size = 15
bar_width = 0.12
bar_zorder = 3
grid = True
# grid_linewidth = 0.2
grid_zorder = 0

dataset = "sin"
# dataset = "twitter"

sin_data_path = {
    "Small Scale": "test_output/sin_smallscale/0318/121432/data/",
    "Middle Scale": "test_output/sin_middlescale/0318/122038/data/",
    "Large Scale": "test_output/sin_largescale/0318/122208/data/",
}

twitter_data_path = {
    "Small Scale": "test_output/twitter_smallscale/0318/121201/data/",
    "Middle Scale": "test_output/twitter_middlescale/0318/121548/data/",
    "Large Scale": "test_output/twitter_largescale/0318/121825/data/"
}

data_path = sin_data_path if dataset == "sin" else twitter_data_path

colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]
# colors = ["#8ECFC9", "#f8c91c", "#FA7F6F", "#376eae"]

""" 1. Averge Request Latency """
def draw_fig1():
    # log
    print(f"Draw Averge Request Latency of {dataset}")
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'Random': [], 'RL Agent': [], 'Proscale': [], 'LGDRL': []}
    for agent in datas.keys():
        for path in data_path.keys():
            data = np.load(os.path.join(data_path[path], agent, "t_all.npy"))
            data = np.mean(data)
            print(f"{agent}: {data}")
            datas[agent].append(data)

    # 设置图表的大小
    fig, ax = plt.subplots(figsize=(7, 5))

    # 图 1: Arrival Rate of Requests
    width = bar_width  # 设置条形图宽度
    x = np.arange(len(data_path.keys()))

    for i, agent in enumerate(datas.keys()):
        ax.bar(x+(i-1.5)*width, datas[agent], width, label=agent, color=colors[i], zorder=bar_zorder)

    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.set_ylabel('Average Request Latency', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(data_path.keys(), fontsize=fontsize)
    ax.legend(frameon=True, fontsize=lengend_fontsize)
    plt.grid(grid, zorder=grid_zorder)
    plt.tight_layout()

    # 显示图表
    save_path = os.path.join(current_dir, f"Averge_Request_Latency-{dataset}.pdf")
    plt.savefig(save_path, format="pdf")

""" 2. Averge Energy Cost """
def draw_fig2():
    # log
    print(f"Draw Averge Energy Cost of {dataset}")

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'Random': [], 'RL Agent': [], 'Proscale': [], 'LGDRL': []}
    for agent in datas.keys():
        for path in data_path.keys():
            data = np.load(os.path.join(data_path[path], agent, "cost.npy"))
            data = np.mean(data)
            print(f"{agent}: {data}")
            datas[agent].append(data)

    # 设置图表的大小
    fig, ax = plt.subplots(figsize=(7, 5))

    width = bar_width  # 设置条形图宽度
    x = np.arange(len(data_path.keys()))

    for i, agent in enumerate(datas.keys()):
        ax.bar(x+(i-1.5)*width, datas[agent], width, label=agent, color=colors[i], zorder=bar_zorder)

    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.set_ylabel('Average Cost', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(data_path.keys(), fontsize=fontsize)
    ax.legend(frameon=True, fontsize=lengend_fontsize)
    plt.grid(grid, zorder=grid_zorder)
    plt.tight_layout()

    # 显示图表
    save_path = os.path.join(current_dir, f"Averge_Cost-{dataset}.pdf")
    plt.savefig(save_path, format="pdf")

""" 3. Averge Request Success Rate """
def draw_fig3():
    # log
    print(f"Draw Averge Request Success Rate of {dataset}")

    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'Random': [], 'RL Agent': [], 'Proscale': [], 'LGDRL': []}
    for agent in datas.keys():
        for path in data_path.keys():
            data = np.load(os.path.join(data_path[path], agent, "request_success_rate.npy"))
            data = np.mean(data)
            print(f"{agent}: {data}")
            datas[agent].append(data)

    # 设置图表的大小
    fig, ax = plt.subplots(figsize=(7, 5))

    width = bar_width  # 设置条形图宽度
    x = np.arange(len(data_path.keys()))

    for i, agent in enumerate(datas.keys()):
        ax.bar(x+(i-1.5)*width, datas[agent], width, label=agent, color=colors[i], zorder=bar_zorder)

    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)
    ax.set_ylabel('Average Cost', fontsize=fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(data_path.keys(), fontsize=fontsize)
    ax.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.grid(grid, zorder=grid_zorder)
    plt.tight_layout()

    # 显示图表
    save_path = os.path.join(current_dir, f"Averge_Request_Success_Rate-{dataset}.pdf")
    plt.savefig(save_path, format="pdf")

if __name__ == '__main__':
    draw_fig1()
    draw_fig2()
    draw_fig3()