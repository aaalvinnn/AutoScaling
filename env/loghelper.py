from environment import DataCenterEnvironment
from collections import defaultdict
from matplotlib import pyplot as plt


class LogHelper(object):
    def __init__(self):
        self.data = defaultdict(list)
        self.ylabel_metrics = {
                        "delay": "ms",
                        "vload": "",
                        "ns": "",
                        "lamda": "requests/hour",
                        "ave_ro": ""
                    }
        self.legend = {
                        "delay": ["total delay","exe delay", "route delay"],
                        "vload": None,
                        "ns": None,
                        "lamda": ["real_lammda", "predicted_lammda"],
                        "ave_ro": None
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


    

