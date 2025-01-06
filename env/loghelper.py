from environment import DataCenterEnvironment
from collections import defaultdict
from matplotlib import pyplot as plt
from env import config


env_config = config.EnvConfig() 
class LogHelper(object):
    def __init__(self):
        self.data = defaultdict(list)
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
                        "ave_lamda": "requests/ts",
                        "lamda": "requests/ts",
                        "ave_ro": "",
                        "request_success_rate": "",
                        "r": "",
                    }
        self.legend = {
                        "t_all": None,
                        "t_exe": None,
                        "t_route": None,
                        "vload": None,
                        "ns": None,
                        "cost": None,
                        "penalty": None,
                        "node_using_num": None,
                        "image_nums": None,
                        "ave_lamda": ["real_lammda", "predicted_lammda"],
                        "lamda": [i for i in range(env_config.ms_nums)],
                        "ave_ro": None,
                        "request_success_rate": None,
                        "r": None,
                    }
        pass

    def record(self, key, value):
        self.data[key].append(value)
        pass

    def visualize(self):
        for key in self.data:
            plt.figure()
            plt.plot(self.data[key], label=key)
            plt.title(f"{key}")
            plt.xlabel("Time Slot")
            plt.ylabel(f"{key}/({self.ylabel_metrics[key]})")
            if self.legend[key]:
                plt.legend(self.legend[key])
            plt.show()

        pass


    

