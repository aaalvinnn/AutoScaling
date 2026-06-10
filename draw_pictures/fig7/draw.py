"""
Fig.7 — Communication Latency Sensitivity Analysis (R2-2)

Communication latency sweep on twitter_largescale:
  T = {1, 2, 3, 4, 5}   BW = D/T (D = avg ms2ms_data = 7.5)

Uses the already-trained LGDRL model. No retraining needed — the observation
space does not contain bandwidth info, so the trained policy makes identical
decisions; only the environment's route-delay calculation changes.

Produces:
  data/bw_{BW}_{t_all,cost,rsr}.npy
  Sensitivity-twitter-LargeScale.pdf
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'env'))

from env import environment
from env.configs import config_twitter_largescale
from methods import PPO_dnn

PPO_dnn.seed_all(1037)

AVG_DATA_SIZE = (5 + 10) / 2  # ms2ms_data uniform [5, 10], mean = 7.5
T_VALUES = [1, 2, 3, 4, 5]
BANDWIDTH_VALUES = [max(0.1, AVG_DATA_SIZE / t) for t in T_VALUES]

MODEL_PATH = os.path.join(
    project_root,
    "model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth",
)

# ── style ────────────────────────────────────────────────────────────
fontsize = 24
legend_fontsize = 18
label_size = 24
line_width = 1.8
markersize = 10
line_zorder = 3
grid_zorder = 0

# ── data collection ──────────────────────────────────────────────────
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)

all_means = {}  # t_val -> [latency, cost, rsr]

for t_val, bw in zip(T_VALUES, BANDWIDTH_VALUES):
    print(f"\n{'='*60}")
    print(f"  T={t_val}  BW={bw} ...")
    print(f"{'='*60}")

    cfg = config_twitter_largescale.EnvConfig()
    cfg.node2node_min_bandwidth = bw
    cfg.node2node_max_bandwidth = bw

    environment.CONFIG = cfg
    agent = PPO_dnn.PPOAgent(cfg)
    agent.load(MODEL_PATH)

    env = environment.DataCenterEnvironment(0, cfg, is_train=False, agent_type="PPO")
    obs, _ = env.reset(seed=1037)

    t_all_list, cost_list, rsr_list = [], [], []
    done = False
    while not done:
        action = agent.get_action(obs)
        obs, reward, done, _, info = env.step(action)
        t_all_list.append(info["t_all"])
        cost_list.append(info["cost"])
        rsr_list.append(info["request_success_rate"])

    arr_t = np.array(t_all_list)
    arr_c = np.array(cost_list)
    arr_r = np.array(rsr_list)

    all_means[t_val] = [np.mean(arr_t), np.mean(arr_c), np.mean(arr_r)]

    np.save(os.path.join(data_dir, f"bw_{bw}_t_all.npy"), arr_t)
    np.save(os.path.join(data_dir, f"bw_{bw}_cost.npy"), arr_c)
    np.save(os.path.join(data_dir, f"bw_{bw}_rsr.npy"), arr_r)

    print(f"  T={t_val} BW={bw}: Lat={np.mean(arr_t):.2f}  Cost={np.mean(arr_c):.2f}  RSR={np.mean(arr_r):.4f}")

# ── summary table ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'T':<6} {'BW':<6} {'Latency':>10} {'Cost':>10} {'RSR':>10}")
print("-" * 65)
for t_val, bw in zip(T_VALUES, BANDWIDTH_VALUES):
    m = all_means[t_val]
    print(f"{t_val:<6} {bw:<6} {m[0]:>10.2f} {m[1]:>10.2f} {m[2]:>10.4f}")
print("=" * 65)

# ── draw: line chart with different markers per metric ───────────────
fig, ax = plt.subplots(figsize=(7, 5))

latency_means = [all_means[t][0] for t in T_VALUES]
cost_means = [all_means[t][1] for t in T_VALUES]
rsr_means = [all_means[t][2] for t in T_VALUES]

def normalize(arr):
    mx = max(arr)
    return [v / mx for v in arr]

ax.plot(T_VALUES, normalize(latency_means), '-o', label="Latency",
        color="#82B0D2", linewidth=line_width, markersize=markersize, zorder=line_zorder)
ax.plot(T_VALUES, normalize(cost_means), '-s', label="Cost",
        color="#FFBE7A", linewidth=line_width, markersize=markersize, zorder=line_zorder)
ax.plot(T_VALUES, normalize(rsr_means), '-^', label="Success Rate",
        color="#FA7F6F", linewidth=line_width, markersize=markersize, zorder=line_zorder)

ax.set_xlabel("Communication Latency T", fontsize=fontsize)
ax.set_ylabel("Normalized Value", fontsize=fontsize)
ax.tick_params(axis='both', labelsize=label_size)
ax.set_xticks(T_VALUES)
ax.legend(frameon=True, fontsize=legend_fontsize, loc='center right')
plt.grid(True, zorder=grid_zorder)
plt.tight_layout()

save_path = os.path.join(current_dir, "Sensitivity-twitter-LargeScale.pdf")
plt.savefig(save_path, format="pdf")
plt.savefig(save_path.replace(".pdf", ".png"), format="png", dpi=150)
plt.close()
print(f"\nSaved: {save_path}")
