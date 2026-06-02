import scienceplots
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os

# 科学绘图包
# plt.style.use(['science','no-latex'])

# 参数设置
fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 2.5
line_zorder = 3
grid_zorder = 0

def read_data(path):
    data = []
    with open(path, "r") as f:
        for line in f.readlines():
            data.append(float(line.strip()))
    return data

""" 1. 正弦、推特数据集到达率示意图 """
if __name__ == '__main__':
    # 获取当前路径
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    file_names = os.listdir("/home/zsw/Works/AutoScaling/data")
    strs = ["Twitter", "Sin"]
    fig, ax = plt.subplots(figsize=(14, 4))
    for i, file in enumerate(file_names):
        if file.endswith(".txt"):
            data = read_data(os.path.join("/home/zsw/Works/AutoScaling/data/", file))
            ax.plot(data, label=strs[i], linewidth=line_width, zorder=line_zorder)
            ax.tick_params(axis='x', labelsize=label_size)
            ax.tick_params(axis='y', labelsize=label_size)
            ax.set_xlabel("Timeslot", fontsize=fontsize)
            ax.set_ylabel(f"Lambda", fontsize=fontsize)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # 强制整数刻度
            ax.grid(True, zorder=grid_zorder)

    plt.legend(frameon=True, fontsize=lengend_fontsize, loc='upper left')
    plt.tight_layout()
    save_path = os.path.join(current_dir, "loads.pdf")
    plt.savefig(save_path, format="pdf")