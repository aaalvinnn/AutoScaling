""" Training convergence — Twitter + Sin + Alibaba """
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

fontsize = 24
lengend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))

def read_sin_training_data(path):
    file_1 = os.path.join(path, "Sin-1.json")
    file_2 = os.path.join(path, "Sin-2.json")
    with open(file_1) as f:
        d1 = np.array(json.load(f))[:, 2]
    with open(file_2) as f:
        d2 = np.array(json.load(f))[:, 2]
    data = np.concatenate((d1, d2[-(20000 - len(d1)):]))
    return -data[::10]

def read_twitter_training_data(path):
    file_path = os.path.join(path, "Twitter.json")
    with open(file_path) as f:
        data = np.array(json.load(f))[:, 2]
    data_2 = data[-(20000 - len(data)):]
    data = np.concatenate((data, data_2))
    return -data[::10]

def read_alibaba_training_data():
    tb_path = os.path.join(ROOT, "model/alibaba_largescale/0602/1440/PPO_dnn")
    ea = EventAccumulator(tb_path)
    ea.Reload()
    events = ea.Scalars('charts/y')
    data = np.array([e.value for e in events])
    return -data

fig, ax = plt.subplots(figsize=(14, 4))
colors = ["#82B0D2", "#8ECFC9", "#FA7F6F"]
json_dir = os.path.join(current_dir, "data", "0603")

tw_data = read_twitter_training_data(json_dir)
sn_data = read_sin_training_data(json_dir)
al_data = read_alibaba_training_data()

ax.plot(np.linspace(0, 20000, len(tw_data)), tw_data, label="Twitter",
        color=colors[0], linewidth=line_width, zorder=line_zorder)
ax.plot(np.linspace(0, 20000, len(sn_data)), sn_data, label="Sin",
        color=colors[1], linewidth=line_width, zorder=line_zorder)
ax.plot(np.linspace(0, 20000, len(al_data)), al_data, label="Alibaba",
        color=colors[2], linewidth=line_width, zorder=line_zorder)

ax.tick_params(axis='both', labelsize=label_size)
ax.set_xlabel("Epoch", fontsize=fontsize)
ax.set_ylabel("Reward", fontsize=fontsize)
ax.grid(True, zorder=grid_zorder)

plt.legend(frameon=True, fontsize=lengend_fontsize, loc='lower left')
plt.tight_layout()

save_path = os.path.join(current_dir, "Train.pdf")
plt.savefig(save_path, format="pdf")
print(f"Saved: {save_path}")
