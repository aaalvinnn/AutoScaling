from environment import DataCenterEnvironment
from collections import defaultdict
from matplotlib import pyplot as plt
from env import config
from datetime import datetime
import os
import numpy as np

env_config = config.EnvConfig() 
class LogHelper(object):
    def __init__(self, agents):
        self.agents_name = agents
        self.data = {f"{agent}": defaultdict(list) for agent in self.agents_name}
        self.ylabel_metrics = {
                        "y": "",
                        "t_all": "ms",
                        "t_exe": "ms",
                        "t_route": "ms",
                        "vload": "",
                        "ns": "",
                        "cost": "",
                        "Qt": "",
                        "penalty": "",
                        "node_using_num": "",
                        "image_nums": "",
                        "predict_lamda": "rts",
                        "lamda": "rts",
                        "lamda_list": "rts",
                        "ave_ro": "",
                        "request_success_rate": "",
                        "r": "",
                    }
        self.save_path = os.path.join("test_output", datetime.now().strftime("%m%d"), datetime.now().strftime("%H%M%S"))
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def record(self, infos):
        if len(infos) != len(self.agents_name):
            raise ValueError(f"The length of infos {len(infos)} is not equal to the length of agents name {self.agents_name}")
        
        for i, agent_name in enumerate(self.agents_name):
            for key in infos[i]:
                self.data[agent_name][key].append(infos[i][key])

    def visualize(self):
        for metric in self.ylabel_metrics.keys():
            if metric == "t_all" or metric == "t_exe" or metric == "t_route" or metric == "request_success_rate" or metric == "y":
                plt.figure(figsize=(10, 6))
                for agent_name in self.agents_name:
                    if metric in self.data[agent_name]:
                        y = moving_average(self.data[agent_name][metric], 9)
                        plt.plot(y, label=agent_name)
                plt.title(f"Comparison of {metric}")
                plt.xlabel("Time Slot")
                plt.ylabel(f"{metric} ({self.ylabel_metrics[metric]})")
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(self.save_path, f"{metric}.png"))

            else:
                plt.figure(figsize=(10, 6))
                for agent_name in self.agents_name:
                    if metric in self.data[agent_name]:
                        plt.plot(self.data[agent_name][metric], label=agent_name)
                plt.title(f"Comparison of {metric}")
                plt.xlabel("Time Slot")
                plt.ylabel(f"{metric} ({self.ylabel_metrics[metric]})")
                plt.legend(loc="best")
                plt.grid(True)
                plt.tight_layout()
                # plt.show()
                plt.savefig(os.path.join(self.save_path, f"{metric}.png"))
    
    def save_data(self):
        for agent_name in self.agents_name:
            for metric in self.data[agent_name]:
                data_save_path = os.path.join(self.save_path, "data", f"{agent_name}", f"{metric}.npy")
                if not os.path.exists(os.path.dirname(data_save_path)):
                    os.makedirs(os.path.dirname(data_save_path))
                np.save(data_save_path, self.data[agent_name][metric])

def moving_average(data, window_size):
    """
    对列表进行移动平均，保持长度不变。
    
    Args:
        data (list or np.ndarray): 输入数据列表。
        window_size (int): 移动窗口大小，必须为奇数。
        
    Returns:
        list: 计算后的移动平均结果，长度与输入相同。
    """
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Input data must be a list or numpy array.")
    if window_size % 2 == 0:
        raise ValueError("Window size must be an odd number.")
    if window_size < 1:
        raise ValueError("Window size must be greater than or equal to 1.")

    # 转换为 numpy 数组
    data = np.array(data)
    half_window = window_size // 2

    # 使用镜像填充边界
    padded_data = np.pad(data, (half_window, half_window), mode='reflect')

    # 计算移动平均
    result = np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')
    return result.tolist()