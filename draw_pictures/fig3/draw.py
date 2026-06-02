import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import json
from matplotlib.ticker import ScalarFormatter

# 科学绘图包
# plt.style.use(['science','no-latex'])

# 参数设置
fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0


def moving_average(data, window_size=3):
    data = data.flatten()
    kernel = np.ones(window_size)/window_size
    data_padded = np.pad(data, (len(kernel)//2, len(kernel)//2), mode='edge')
    return np.convolve(data_padded, kernel, mode='valid')

def read_sin_training_data(path):
    file_1_path = os.path.join(path, "Sin-1.json")
    file_2_path = os.path.join(path, "Sin-2.json")
    with open(file_1_path, "r") as f:
        data_1 = json.load(f)
        data_1 = np.array(data_1)[:,2]
    with open(file_2_path, "r") as f:
        data_2 = json.load(f)
        data_2 = np.array(data_2)[:,2]

    data = np.concatenate((data_1, data_2[-(20000-len(data_1)):]))

    data = data[::10]
    return -data

def read_twitter_training_data(path):
    file_path = os.path.join(path, "Twitter.json")
    with open(file_path, "r") as f:
        data = json.load(f)
        data = np.array(data)[:,2]

    data_2 = data[-(20000-len(data)):]
    data = np.concatenate((data, data_2))
    data = data[::10]
    return -data



""" 1. 正弦、推特数据集训练收敛图示意图 """
if __name__ == '__main__':
    # 获取当前路径
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    strs = ["Twitter", "Sin"]

    fig, ax = plt.subplots(figsize=(14, 4))

    x = np.arange(0, 10000, 10)

    # 读取数据
    for str in strs:
        file_path = os.path.join(current_dir, "data", "0603")
        if str == "Twitter":
            data = read_twitter_training_data(file_path)
        else:
            data = read_sin_training_data(file_path)

        ax.plot(data, label=str, linewidth=line_width, zorder=line_zorder)
        ax.tick_params(axis='x', labelsize=label_size)
        ax.tick_params(axis='y', labelsize=label_size)
        ax.set_xlabel("Epoch", fontsize=fontsize)
        ax.set_ylabel(f"Reward", fontsize=fontsize)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 强制整数刻度
        ax.grid(True, zorder=grid_zorder)

    # 设置科学计数法格式
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))  # 控制科学计数法的阈值
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.tick_params(axis='both', labelsize=label_size)

    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='lower left')
    plt.tight_layout()
    save_path = os.path.join(current_dir, "Train.pdf")
    plt.savefig(save_path, format="pdf")