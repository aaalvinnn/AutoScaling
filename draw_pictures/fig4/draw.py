import scienceplots
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager


# 参数设置
fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0
alpha = 1
markersize = 2.5

# plt.style.use(['science','no-latex'])
# prop = font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')


dataset = "sin"
# dataset = "twitter"

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

data_path = sin_data_path if dataset == "sin" else twitter_data_path

scale = "Large Scale"

# colors = ["#8ECFC9", "#FFBE7A", "#FA7F6F", "#82B0D2"]
# colors = ["#62BEA6", "#FDBA6B", "#EB6046", "#4B66AD"]
# colors = ["#", "#c22f2f", "#449945", "#354e6b"]
# colors = ["#449945", "#ea7827", "#c22f2f", "#354e6b"]
colors = ["#449945", "#f8c91c", "#376eae", "#c22f2f"]

def moving_average(data, window_size=3):
    kernel = np.ones(window_size)/window_size
    data_padded = np.pad(data, (len(kernel)//2, len(kernel)//2), mode='edge')
    return np.convolve(data_padded, kernel, mode='valid')

""" 1. Averge Request Latency """
def draw_fig1_v1():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'LGDRL': None}
    for agent in datas.keys():
        data = np.load(os.path.join(data_path[scale], agent, "t_all.npy"))[20:]
        # data = data[::6]
        # data = moving_average(data, window_size=5)
        datas[agent] = data

    # 设置图表的大小
    plt.figure(figsize=(7, 5))

    for i, agent in enumerate(datas.keys()):
        if agent == "LGDRL":
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder)
        else:
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder, alpha=alpha)

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Latency', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper right')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Latency-{dataset}-{scale}.pdf")
    plt.savefig(save_path, format="pdf")

def draw_fig1():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'LGDRL': None}
    strs = ['RL Agent', 'Proscale', 'PPA', 'Ours']
    for agent in datas.keys():
        data = np.load(os.path.join(data_path[scale], agent, "t_all.npy"))
        # data = data[::6]
        # data = moving_average(data, window_size=5)
        datas[agent] = data

    # 设置图表的大小
    plt.figure(figsize=(14, 4))

    for i, agent in enumerate(datas.keys()):
        plt.plot(datas[agent], label=strs[i], color=colors[i], linewidth=line_width, zorder=line_zorder, linestyle='--', marker='o', markersize=markersize)


    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Latency', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper right')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Latency-{dataset}-{scale}.pdf")
    plt.savefig(save_path, format="pdf")

""" 2. Averge Energy Cost """
def draw_fig2():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'LGDRL': None}
    strs = ['RL Agent', 'Proscale', 'PPA', 'Ours']
    for agent in datas.keys():
        datas[agent] = np.load(os.path.join(data_path[scale], agent, "cost.npy"))

    # 设置图表的大小
    plt.figure(figsize=(7, 5))

    for i, agent in enumerate(datas.keys()):
        plt.plot(datas[agent], label=strs[i], color=colors[i], linewidth=line_width, zorder=line_zorder)


    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    if dataset == "sin":
        plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper right')
    else:
        plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Cost-{dataset}-{scale}.pdf")
    plt.savefig(save_path, format="pdf")

""" 3. Averge Request Success Rate """
def draw_fig3():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    smooth_window_size = 9
    datas = {'RL Agent': None, 'Proscale': None, 'HPA': None, 'LGDRL': None}
    for agent in datas.keys():
        datas[agent] = np.load(os.path.join(data_path[scale], agent, "request_success_rate.npy"))
        datas[agent] = np.convolve(datas[agent], np.ones(smooth_window_size) / smooth_window_size, mode='valid')

    # 设置图表的大小
    plt.figure(figsize=(7, 5))

    for i, agent in enumerate(datas.keys()):
        if agent == "LGDRL":
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder)
        else:
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder, alpha=alpha)

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Success Rate', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='lower left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Success_Rate-{dataset}-{scale}.pdf")
    plt.savefig(save_path, format="pdf")

if __name__ == '__main__':
    draw_fig1()
    draw_fig2()
    # draw_fig3()