""" 20 节点 (twitter_xlargescale) 三方法训练收敛曲线 —— fig3 风格。

从各方法 TensorBoard 日志读 charts/y（Lyapunov drift，越低越好），取负当 reward，
画 LGDRL(PPO) / DeepScaler / RL Agent(SAC) 的 epoch-reward 收敛曲线，
并把逐 epoch 数据存成 JSON。输出到 draw_pictures/fig3/。

注意：本机 tensorboard 2.20 的 C reader 在 numpy 2.2/1.26 下都会 nondeterministic
内存腐蚀（int/str 类型错乱），故这里用纯 Python 手写 tfrecord+protobuf 解析器绕过它。
"""
import json
import os
import struct
import glob

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
RUN_DIR = os.path.join(ROOT, "model", "twitter_xlargescale", "0619", "2040")

# 三方法的 TensorBoard 目录、图例标签、颜色、线型（颜色与 fig4 一致）
methods = [
    {"tb": os.path.join(RUN_DIR, "PPO_dnn"),    "label": "Ours (LGDRL)",   "color": "#c22f2f", "ls": "-."},
    {"tb": os.path.join(RUN_DIR, "DeepScaler"), "label": "DeepScaler",     "color": "#8E6EC8", "ls": "--"},
    {"tb": os.path.join(RUN_DIR, "SAC"),        "label": "RL Agent (SAC)", "color": "#449945", "ls": "--"},
]

SMOOTH_WIN = 200   # 收敛曲线滚动平滑窗（原始数据仍存 JSON）


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
            yield fnum, wire, buf[i:i+8]; i += 8
        elif wire == 2:
            ln, i = _read_varint(buf, i); yield fnum, wire, buf[i:i+ln]; i += ln
        elif wire == 5:
            yield fnum, wire, buf[i:i+4]; i += 4
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
    """读目录下所有 events.out.tfevents.* 的 charts/y，返回 (steps, values)，重复 step 后写覆盖。"""
    by_step = {}
    for path in sorted(glob.glob(os.path.join(d, "events.out.tfevents.*"))):
        buf = open(path, "rb").read()
        i, n = 0, len(buf)
        while i + 12 <= n:
            length = struct.unpack("<Q", buf[i:i+8])[0]
            i += 12  # 8 字节 length + 4 字节 length_crc
            if i + length + 4 > n:
                break
            data = buf[i:i+length]; i += length + 4  # data + 4 字节 data_crc
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


def smooth(x, win):
    if win <= 1 or len(x) < win:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="valid")


# ── 绘图 ──────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 4))
payload = {"config": "twitter_xlargescale", "methods": [m["label"] for m in methods],
           "per_method": {}, "summary": {}}

for m in methods:
    it, y = read_tfevents_dir(m["tb"])
    rew = -y  # 取负当 reward（y 越低越好 → reward 越高越好）
    payload["per_method"][m["label"]] = {"epoch": it.tolist(), "reward": rew.tolist()}
    tail = rew[max(0, int(len(rew) * 0.9)):]
    payload["summary"][m["label"]] = {
        "num_epochs": int(len(rew)),
        "final_reward_mean": float(np.mean(tail)),
        "final_reward_std": float(np.std(tail)),
    }
    sm = smooth(rew, SMOOTH_WIN)
    ax.plot(it[:len(sm)], sm, label=m["label"], color=m["color"],
            linewidth=line_width, zorder=line_zorder, linestyle=m["ls"])

ax.tick_params(axis="both", labelsize=label_size)
ax.set_xlabel("Epoch", fontsize=fontsize)
ax.set_ylabel("Reward", fontsize=fontsize)
ax.grid(True, zorder=grid_zorder)
plt.legend(frameon=True, fontsize=legend_fontsize, loc="lower right")
plt.tight_layout()

pdf_path = os.path.join(current_dir, "Convergence-twitter-xlarge.pdf")
plt.savefig(pdf_path, format="pdf")
plt.close()

json_path = os.path.join(current_dir, "data_convergence_twitter_xlargescale.json")
with open(json_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"PDF  → {pdf_path}")
print(f"JSON → {json_path}")
print(f"\n收敛值（末 10% epoch 平均 reward = -Lyapunov y，越高越好）:")
for m in methods:
    s = payload["summary"][m["label"]]
    print(f"  {m['label']:18s} epochs={s['num_epochs']:5d}  reward={s['final_reward_mean']:8.2f}±{s['final_reward_std']:.2f}")
