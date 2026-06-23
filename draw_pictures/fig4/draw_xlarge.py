""" 20 节点逐时隙延迟与开销曲线 —— fig4 风格（twitter + alibaba _deep）。

读 test_output/{config}/data/{agent}/t_all.npy 与 cost.npy（由 eval_xlarge_deep.py 生成），
画 Latency (14x4) 与 Cost (7x5) 折线图 + 存 JSON，输出到 draw_pictures/fig4/。
LGDRL = 4 层 _deep 模型，DeepScaler = model_1000（与 10 节点约定一致）。
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# ── 样式（与 fig4/draw.py 完全一致）───────────────────────────────────────────
fontsize = 24
legend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0
markersize = 2.5

scale = "XLarge Scale"
# (config_name, dataset 标签) —— 两个 20 节点 _deep 场景
CONFIGS = [
    ("twitter_xlargescale_deep", "twitter"),
    ("alibaba_xlargescale_deep", "alibaba"),
]

# 5 个对比算法（顺序、颜色、线型、图例标签 与 fig4 一致）
agents = ['RL Agent', 'Proscale', 'HPA', 'DeepScaler', 'LGDRL']
labels = ['RL Agent', 'Proscale', 'PPA', 'DeepScaler', 'Ours']
colors = ["#449945", "#f8c91c", "#376eae", "#8E6EC8", "#c22f2f"]
linestyles = ['--', '--', '--', '--', '-.']

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))


def _data_dir(config_name):
    # 本地 data/ 副本（原始 per-step npy 由 eval_xlarge_deep.py 生成到 test_output/）
    return os.path.join(current_dir, "data", config_name)


def _load(config_name, metric):
    dp = _data_dir(config_name)
    return {a: np.load(os.path.join(dp, a, f"{metric}.npy")) for a in agents}


def draw_latency(config_name, dataset):
    datas = _load(config_name, "t_all")
    plt.figure(figsize=(14, 4))
    for i, a in enumerate(agents):
        plt.plot(datas[a], label=labels[i], color=colors[i], linewidth=line_width,
                 zorder=line_zorder, linestyle=linestyles[i], marker='o', markersize=markersize)
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Latency', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='upper right')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    out = os.path.join(current_dir, f"Latency-{dataset}-{scale}.pdf")
    plt.savefig(out, format="pdf")
    plt.close()
    return out


def draw_cost(config_name, dataset):
    datas = _load(config_name, "cost")
    plt.figure(figsize=(7, 5))
    for i, a in enumerate(agents):
        plt.plot(datas[a], label=labels[i], color=colors[i], linewidth=line_width,
                 zorder=line_zorder, linestyle=linestyles[i])
    plt.xticks(fontsize=label_size)
    plt.yticks(fontsize=label_size)
    plt.ylabel('Cost', fontsize=fontsize)
    plt.xlabel("Timeslot", fontsize=fontsize)
    plt.legend(frameon=True, fontsize=legend_fontsize, loc='upper left')
    plt.tight_layout()
    plt.grid(True, zorder=grid_zorder)
    out = os.path.join(current_dir, f"Cost-{dataset}-{scale}.pdf")
    plt.savefig(out, format="pdf")
    plt.close()
    return out


def dump_json(config_name, dataset):
    t_all = _load(config_name, "t_all")
    cost = _load(config_name, "cost")
    payload = {
        "config": config_name, "scale": scale,
        "steps": int(next(iter(t_all.values())).shape[0]),
        "agents": labels,
        "per_step": {labels[i]: {"t_all": t_all[a].tolist(), "cost": cost[a].tolist()}
                     for i, a in enumerate(agents)},
        "summary": {labels[i]: {"t_all_mean": float(t_all[a].mean()), "t_all_std": float(t_all[a].std()),
                                "cost_mean": float(cost[a].mean()), "cost_std": float(cost[a].std())}
                    for i, a in enumerate(agents)},
    }
    os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)
    out = os.path.join(current_dir, "data", f"data_{config_name}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


if __name__ == '__main__':
    for config_name, dataset in CONFIGS:
        dp = _data_dir(config_name)
        assert os.path.isdir(dp), f"数据目录不存在: {dp}\n先跑 eval_xlarge_deep.py 生成 per-step npy"
        p1 = draw_latency(config_name, dataset)
        p2 = draw_cost(config_name, dataset)
        p3 = dump_json(config_name, dataset)
        print(f"[{dataset}] Latency → {p1}")
        print(f"[{dataset}] Cost    → {p2}")
        print(f"[{dataset}] JSON    → {p3}")
        for a, s in json.load(open(p3))["summary"].items():
            print(f"  {a:12s} latency={s['t_all_mean']:6.2f}±{s['t_all_std']:.2f}  cost={s['cost_mean']:6.2f}±{s['cost_std']:.2f}")
        print()
