""" 20 节点 (twitter_xlargescale) 逐时隙延迟与开销曲线 —— fig4 风格。

读 test_output/twitter_xlargescale/data/{agent}/t_all.npy 与 cost.npy，
画 Latency (14x4) 与 Cost (7x5) 折线图，并把逐 step 数据 + 汇总统计存成 JSON，
全部输出到 draw_pictures/fig4/。
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

dataset = "twitter"
scale = "XLarge Scale"                       # 20 节点场景标签
config_name = "twitter_xlargescale"

# 5 个对比算法（顺序、颜色、线型、图例标签 与 fig4 一致）
agents = ['RL Agent', 'Proscale', 'HPA', 'DeepScaler', 'LGDRL']
labels = ['RL Agent', 'Proscale', 'PPA', 'DeepScaler', 'Ours']
colors = ["#449945", "#f8c91c", "#376eae", "#8E6EC8", "#c22f2f"]
linestyles = ['--', '--', '--', '--', '-.']

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))
data_path = os.path.join(ROOT, "test_output", config_name, "data")


def _load(metric):
    return {a: np.load(os.path.join(data_path, a, f"{metric}.npy")) for a in agents}


def draw_latency():
    datas = _load("t_all")
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


def draw_cost():
    datas = _load("cost")
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


def dump_json():
    """逐 step 数据 + 汇总统计存 JSON。"""
    t_all = _load("t_all")
    cost = _load("cost")
    payload = {
        "config": config_name,
        "scale": scale,
        "steps": int(next(iter(t_all.values())).shape[0]),
        "agents": labels,
        "per_step": {
            labels[i]: {
                "t_all": t_all[a].tolist(),
                "cost": cost[a].tolist(),
            } for i, a in enumerate(agents)
        },
        "summary": {
            labels[i]: {
                "t_all_mean": float(t_all[a].mean()), "t_all_std": float(t_all[a].std()),
                "cost_mean": float(cost[a].mean()), "cost_std": float(cost[a].std()),
            } for i, a in enumerate(agents)
        },
    }
    out = os.path.join(current_dir, f"data_{config_name}.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


if __name__ == '__main__':
    assert os.path.isdir(data_path), f"数据目录不存在: {data_path}\n先跑: python main.py --config " + config_name
    p1 = draw_latency()
    p2 = draw_cost()
    p3 = dump_json()
    print(f"Latency → {p1}")
    print(f"Cost    → {p2}")
    print(f"JSON    → {p3}")
    print("\n汇总 (mean ± std):")
    import json as _j
    d = _j.load(open(p3))
    for a, s in d["summary"].items():
        print(f"  {a:12s} latency={s['t_all_mean']:6.2f}±{s['t_all_std']:.2f}  cost={s['cost_mean']:6.2f}±{s['cost_std']:.2f}")
