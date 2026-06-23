""" 20 节点 twitter Ours(LGDRL-4L, _deep, 20000 epoch) reward 收敛曲线 —— fig3 风格。

读 model/twitter_xlargescale_deep/0621/1617/PPO_dnn 的 TensorBoard charts/y
（Lyapunov drift，越低越好），取负当 reward，画单条 Ours 收敛曲线（淡色原始 + 粗实线平滑），
逐 epoch 数据存 JSON。输出到 draw_pictures/fig3/。

纯 Python 手写 tfevents+protobuf 解析器（绕开本机 tensorboard C reader 的内存腐蚀）。
"""
import glob
import json
import os
import struct

import matplotlib.pyplot as plt
import numpy as np

# ── 样式（与 fig3/draw.py 一致）───────────────────────────────────────────────
fontsize = 24
legend_fontsize = 21
label_size = 24
line_width = 1.5
line_zorder = 3
grid_zorder = 0

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))
TB_DIR = os.path.join(ROOT, "model", "twitter_xlargescale_deep", "0621", "1617", "PPO_dnn")


# ── 纯 Python tfevents 解析器（绕开 tensorboard 的腐蚀 C reader）─────────────────
def _read_varint(buf, i):
    result = shift = 0
    while True:
        b = buf[i]; i += 1
        result |= (b & 0x7f) << shift
        if not (b & 0x80):
            return result, i
        shift += 7


def _iter_fields(buf):
    i, n = 0, len(buf)
    while i < n:
        tag, i = _read_varint(buf, i)
        fnum, wire = tag >> 3, tag & 7
        if wire == 0:
            v, i = _read_varint(buf, i); yield fnum, wire, v
        elif wire == 1:
            yield fnum, wire, buf[i:i + 8]; i += 8
        elif wire == 2:
            ln, i = _read_varint(buf, i); yield fnum, wire, buf[i:i + ln]; i += ln
        elif wire == 5:
            yield fnum, wire, buf[i:i + 4]; i += 4
        else:
            break


def _parse_value(vb):
    tag = sv = None
    for fnum, wire, v in _iter_fields(vb):
        if fnum == 1 and wire == 2:
            tag = v.decode("utf-8", "replace")
        elif fnum == 2 and wire == 5:
            sv = struct.unpack("<f", v)[0]
    return tag, sv


def read_tfevents_dir(d, want="charts/y"):
    by_step = {}
    for path in sorted(glob.glob(os.path.join(d, "events.out.tfevents.*"))):
        buf = open(path, "rb").read()
        i, n = 0, len(buf)
        while i + 12 <= n:
            length = struct.unpack("<Q", buf[i:i + 8])[0]
            i += 12
            if i + length + 4 > n:
                break
            data = buf[i:i + length]; i += length + 4
            try:
                step = summary = None
                for fnum, wire, v in _iter_fields(data):
                    if fnum == 2 and wire == 0:
                        step = v
                    elif fnum == 5 and wire == 2:
                        summary = v
                if step is None or summary is None:
                    continue
                for fnum, wire, v in _iter_fields(summary):
                    if fnum == 1 and wire == 2:
                        tag, sv = _parse_value(v)
                        if tag == want and sv is not None:
                            by_step[step] = sv
            except Exception:
                continue
    steps = np.array(sorted(by_step), dtype=np.int64)
    vals = np.array([by_step[s] for s in steps], dtype=np.float64)
    return steps, vals


# ── 画图代码直接照搬 fig3/draw.py（不平滑，仅 [::10] 抽稀 + linspace 横轴）──────
it, y = read_tfevents_dir(TB_DIR)
rew = -y                  # 与 draw.py 的 -data 等价：取负当 reward
data = rew[::10]          # 与 draw.py 的 data[::10] 一致：每 10 点取 1，不做任何平滑

tail = rew[max(0, int(len(rew) * 0.9)):]
summary = {"num_epochs": int(len(rew)),
           "final_reward_mean": float(np.mean(tail)),
           "final_reward_std": float(np.std(tail))}

fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(np.linspace(0, 20000, len(data)), data, label="Ours (LGDRL)",
        color="#c22f2f", linewidth=line_width, zorder=line_zorder)

ax.tick_params(axis="both", labelsize=label_size)
ax.set_xlabel("Epoch", fontsize=fontsize)
ax.set_ylabel("Reward", fontsize=fontsize)
ax.grid(True, zorder=grid_zorder)
plt.legend(frameon=True, fontsize=legend_fontsize, loc="lower left")
plt.tight_layout()

pdf_path = os.path.join(current_dir, "Convergence-twitter-xlarge-deep.pdf")
plt.savefig(pdf_path, format="pdf")
plt.close()

# 存成 JSON：横轴用与图一致的 linspace，reward 为抽稀后的原始值（无平滑）
payload = {"config": "twitter_xlargescale_deep", "method": "Ours (LGDRL)",
           "epoch": np.linspace(0, 20000, len(data)).tolist(),
           "reward": data.tolist(), "summary": summary}
json_path = os.path.join(current_dir, "data_convergence_twitter_xlarge_deep.json")
with open(json_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"PDF  → {pdf_path}")
print(f"JSON → {json_path}")
print(f"\nOurs (LGDRL)  epochs={summary['num_epochs']}  "
      f"末10% reward={summary['final_reward_mean']:.2f}±{summary['final_reward_std']:.2f}")
print(f"reward 范围: {rew.min():.2f} → {rew.max():.2f}")
