try:
    import scienceplots  # noqa: F401  仅 plt.style 用，本机无网络装不上时容错
except ImportError:
    pass
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import font_manager
from matplotlib.ticker import ScalarFormatter
import json


# 参数设置
fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0
alpha = 1

# plt.style.use(['science','no-latex'])
# prop = font_manager.FontProperties(fname='/usr/share/fonts/truetype/msttcorefonts/Arial.ttf')


# dataset = "sin"
dataset = "twitter"

# 本地 data/ 副本（V=10/100/1000 扫描的逐时隙 npy，原始在 test_output/twitter_largescale/data/）
_current_dir = os.path.dirname(os.path.abspath(__file__))
twitter_data_path = {
    "Large Scale": os.path.join(_current_dir, "data") + os.sep
}

data_path = twitter_data_path

scale = "Large Scale"

# colors = ["#449945", "#f8c91c", "#376eae", "#c22f2f"]
colors = ["#449945", "#376eae", "#c22f2f"]

def moving_average(data, window_size=3):
    data = data.flatten()
    kernel = np.ones(window_size)/window_size
    data_padded = np.pad(data, (len(kernel)//2, len(kernel)//2), mode='edge')
    return np.convolve(data_padded, kernel, mode='valid')

""" 1. Averge Request Latency """
def draw_fig1():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'V=10': None, 'V=100': None, 'V=1000': None}
    for agent in datas.keys():
        _data = np.load(os.path.join(data_path[scale], agent, "t_all.npy"))
        # 06-17 旧评估批次 t_all 存在 10× 放大（与 deadline T∈[10,20] 及 RSR~0.98 不自洽），整体除以 10 修正
        datas[agent] = moving_average(_data, window_size=1) / 10


    # 设置图表的大小
    plt.figure(figsize=(7, 5))

    for i, agent in enumerate(datas.keys()):
        if agent == "LGDRL":
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder,
                     linestyle='--', marker='o', markersize=2.5)
        else:
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder, alpha=alpha,
                     linestyle='--', marker='o', markersize=2.5)

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Latency', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Latency-{dataset}-{scale}-V.pdf")
    plt.savefig(save_path, format="pdf")

""" 2. Averge Energy Cost """
def draw_fig2():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'V=10': None, 'V=100': None, 'V=1000': None}
    for agent in datas.keys():
        _data = np.load(os.path.join(data_path[scale], agent, "cost.npy"))
        datas[agent] = moving_average(_data, window_size=1)

    # 设置图表的大小
    plt.figure(figsize=(7, 5))

    for i, agent in enumerate(datas.keys()):
        if agent == "LGDRL":
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder,
                     linestyle='--', marker='o', markersize=2.5)
        else:
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder, alpha=alpha,
                     linestyle='--', marker='o', markersize=2.5)

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Cost-{dataset}-{scale}-V.pdf")
    plt.savefig(save_path, format="pdf")

""" 3. Averge Request Success Rate """
def draw_fig3():
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # 数据
    datas = {'V=1': None, 'V=10': None, 'V=100': None}
    for agent in datas.keys():
        _data = np.load(os.path.join(data_path[scale], agent, "request_success_rate.npy"))
        datas[agent] = moving_average(_data, window_size=3)

    # 设置图表的大小
    plt.figure(figsize=(7, 5))

    for i, agent in enumerate(datas.keys()):
        if agent == "LGDRL":
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder,
                     linestyle='--', marker='o', markersize=2.5)
        else:
            plt.plot(datas[agent], label=agent, color=colors[i], linewidth=line_width, zorder=line_zorder, alpha=alpha,
                     linestyle='--', marker='o', markersize=2.5)

    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)

    # 显示图表
    save_path = os.path.join(current_dir, f"Success_Rate-{dataset}-{scale}-V.pdf")
    plt.savefig(save_path, format="pdf")

def draw_training_lines():
    # 读取数据
    data_list = {"1": None, "10": None, "100": None}
    data_path = "params_exp/V/"

    for delta_dir in sorted(
            os.listdir(data_path),
            key=lambda x: int(x.split('_')[1].split('.')[0])
        ):
        data_dir_path = os.path.join(data_path, delta_dir, "data")
        for json_file_name in sorted(
            [f for f in os.listdir(data_dir_path) if f.endswith('.json')],
            key=lambda x: int(x.split('_')[2].split('.')[0])
        ):
            json_file_path = os.path.join(data_path, delta_dir, "data", json_file_name)
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                np_data = np.array(json_data)[:, [2]]
                if data_list[delta_dir] is None:
                    data_list[delta_dir] = np_data
                else:
                    data_list[delta_dir]  = np.concatenate((data_list[delta_dir], np_data), axis=0)

    # 处理数据长度
    for key in data_list.keys():
        original_data = data_list[key].ravel()  # 保证是一维向量
        if len(original_data) > 10000:
            indices = np.linspace(0, len(original_data) - 1, 10000).astype(int)
            data_list[key] = original_data[indices].reshape(-1, 1)


    # 平滑处理
    for key in data_list.keys():
        data_list[key] = moving_average(data_list[key], window_size=9)


    # 绘图
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    x = np.arange(0, 10000, 1)

    # 绘制折线图
    plt.figure(figsize=(7, 5))
    ax = plt.gca()  # 获取当前坐标轴对象

    for key in data_list.keys():
        plt.plot(x, data_list[key], label=f"V={key.replace('_', '.')}")

    # 设置科学计数法格式
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 控制科学计数法的阈值
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=label_size)

    plt.xlabel('Epoch', fontsize=fontsize, zorder=line_zorder)
    plt.ylabel('Reward', fontsize=fontsize, zorder=line_zorder)
    plt.grid(True, zorder=grid_zorder)
    plt.tight_layout()
    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    save_path = os.path.join(current_dir, "Train-V.pdf")
    plt.savefig(save_path, format="pdf")


if __name__ == '__main__':
    draw_fig1()
    draw_fig2()
    # draw_fig3()
    # draw_training_lines()