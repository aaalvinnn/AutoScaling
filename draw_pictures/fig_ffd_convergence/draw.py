""" 意见 6 —— 部署/路由解耦消融：MFFD 初始部署 vs 随机初始部署的收敛对比（alibaba_largescale）。

读两份 PPO 训练 TensorBoard 日志的 charts/y（Lyapunov drift，越低越好，取负当 reward）：
  - Full（MFFD 初始部署）       model/alibaba_largescale/0602/1440/PPO_dnn
  - w/o FFD（随机初始部署，重训） model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn

出图：两条平滑后的 epoch–reward 收敛曲线（MFFD 收敛更快/更高 = 部署质量帮助收敛）。
并打印：末 10% epoch 平均 reward（收敛效果）+ 首次达到末值 95% 的 epoch（收敛时间）。

纯 Python 手写 tfevents 解析器，绕开本机 tensorboard C reader 在 numpy 2.2 下的腐蚀。
"""
import json
import os
import struct
import glob

import matplotlib.pyplot as plt
import numpy as np

# ── 样式（与 fig3 一致）──────────────────────────────────────────────────────
fontsize = 24
legend_fontsize = 21
label_size = 24
line_width = 1.8
line_zorder = 3
grid_zorder = 0

current_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(current_dir))

# 两方法的 TB 目录、图例、颜色（颜色与 fig6 一致：Full 红，w/o FFD 绿）
RUN_DIR = ROOT

# DATASET 开关：alibaba / twitter。两份对照训练的 TB 目录各一组。
DATASETS = {
    "alibaba": {
        "full":   "model/alibaba_largescale/0602/1440/PPO_dnn",
        "no_ffd": "model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn",
    },
    "twitter": {
        "full":   "model/twitter_largescale/0530/1829/PPO_dnn",
        "no_ffd": "model/twitter_largescale_no_ffd/0622/1143/PPO_dnn",
    },
}
DATASET = "twitter"   # ← 切 alibaba / twitter

_runs = DATASETS[DATASET]
methods = [
    {"tb": os.path.join(RUN_DIR, _runs["full"]),   "label": "AutoLFD (MFFD init)",   "color": "#c22f2f", "ls": "-"},
    {"tb": os.path.join(RUN_DIR, _runs["no_ffd"]), "label": "w/o FFD (random init)", "color": "#449945", "ls": "--"},
]

SMOOTH_WIN = 200   # 收敛曲线滚动平滑窗（原始数据仍存 JSON）


# ── 纯 Python tfevents 解析器（与 fig3/draw_xlarge_methods.py 同实现）──────────
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
    """读目录下所有 events.out.tfevents.* 的 want tag，返回 (steps, values)，重复 step 后写覆盖。"""
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
payload = {"config": f"{DATASET}_largescale", "comparison": "MFFD vs random initial deployment",
           "per_method": {}, "summary": {}}

for m in methods:
    it, y = read_tfevents_dir(m["tb"], want="charts/y")
    rew = -y  # y 越低越好 → reward 越高越好
    payload["per_method"][m["label"]] = {"epoch": it.tolist(), "reward": rew.tolist()}

    tail = rew[max(0, int(len(rew) * 0.9)):]
    final_mean = float(np.mean(tail))
    # 收敛时间：平滑后首次完成「总提升量 90%」的 epoch
    #   start 取前 1% 平滑值（避开第 0 步抖动），final 取末 10% 均值
    sm_all = smooth(rew, min(SMOOTH_WIN, len(rew)))
    it_sm = it[:len(sm_all)]
    start_mean = float(np.mean(sm_all[:max(1, len(sm_all) // 100)]))
    threshold = start_mean + 0.90 * (final_mean - start_mean)   # reward 越大越好
    hit = sm_all >= threshold
    conv_epoch = int(it_sm[np.argmax(hit)]) if np.any(hit) else int(it[-1])

    payload["summary"][m["label"]] = {
        "num_epochs": int(len(rew)),
        "final_reward_mean": final_mean,
        "final_reward_std": float(np.std(tail)),
        "conv_epoch_90pct_improve": conv_epoch,
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

pdf_path = os.path.join(current_dir, f"Convergence-FFD-ablation-{DATASET}.pdf")
plt.savefig(pdf_path, format="pdf")
plt.savefig(pdf_path.replace(".pdf", ".png"), format="png", dpi=150)
plt.close()

# JSON 落进 data/（与逐 epoch 原始数据同放，便于 release 复现）
os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)
json_path = os.path.join(current_dir, "data", f"data_convergence_ffd_ablation_{DATASET}.json")
with open(json_path, "w") as f:
    json.dump(payload, f, indent=2)

print(f"PDF  → {pdf_path}")
print(f"JSON → {json_path}")
print(f"\n收敛对比（末 10% epoch 平均 reward = -Lyapunov y，越高越好；conv = 平滑后首达总提升 90% 的 epoch）:")
for m in methods:
    s = payload["summary"][m["label"]]
    print(f"  {m['label']:26s} epochs={s['num_epochs']:6d}  reward={s['final_reward_mean']:9.2f}±{s['final_reward_std']:.2f}  conv@90%={s['conv_epoch_90pct_improve']}")
