from environment import DataCenterEnvironment
from collections import defaultdict
from matplotlib import pyplot as plt
from env import config


env_config = config.EnvConfig() 
class LogHelper(object):
    def __init__(self, agents):
        self.agents_name = agents
        self.data = {f"{agent}": defaultdict(list) for agent in self.agents_name}
        self.ylabel_metrics = {
                        "t_all": "ms",
                        "t_exe": "ms",
                        "t_route": "ms",
                        "vload": "",
                        "ns": "",
                        "cost": "",
                        "penalty": "",
                        "node_using_num": "",
                        "image_nums": "",
                        "predict_lamda": "rts",
                        "lamda": "rts",
                        "lamda": "requests/ts",
                        "ave_ro": "",
                        "request_success_rate": "",
                        "r": "",
                    }

    def record(self, infos):
        if len(infos) != len(self.agents_name):
            raise ValueError(f"The length of infos {len(infos)} is not equal to the length of agents name {self.agents_name}")
        
        for i, agent_name in enumerate(self.agents_name):
            for key in infos[i]:
                self.data[agent_name][key].append(infos[i][key])

    def visualize(self):
        for metric in self.ylabel_metrics.keys():
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
            plt.show()