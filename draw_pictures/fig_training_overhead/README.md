# Fig.Training-Overhead — 训练与决策开销（R2-5）

> 对应审稿意见 **R2-5**（P0）：审稿人指出 "The paper does not discuss the training overhead or convergence time of the MDRL approach"，要求补充训练收敛时间与计算开销。本目录产出 10 节点 Twitter 场景的训练开销表（主表），并在 Markdown 里附 20 节点对照行。

## 图片说明 / 产出

本图以**表格**形式呈现（非折线/柱状图），用于论文 5.2 节。指标：训练 epoch、收敛 epoch、收敛/总训练 wall time、SPS、单步决策延迟（mean/p95）、模型大小、参数量。

## 生成文件

| 文件 | 内容 |
|------|------|
| `training_overhead_10node.md` | 10 节点主表 + 20 节点对照行（Markdown，人读） |
| `training_overhead_10node.csv` | 同主表（CSV） |
| `training_overhead_10node.tex` | 同主表（LaTeX，可直接进论文） |
| `data/overhead_10node.json` | 完整结构化数据（含收敛规则元数据、20 节点 extra 行） |
| `data/decision_latency_10node.npy` | 10 节点三方法逐步决策延迟数组（dict） |

## 脚本

- `collect_overhead.py` — 自包含采集器：
  1. 纯 Python tfevents 解析器读 `charts/y` + `charts/SPS`（绕开 tensorboard C reader 腐蚀），算收敛 epoch/时间；
  2. 子进程加载各方法 checkpoint 算参数量/模型大小；
  3. 计时 `agent.get_action(state)` 测决策延迟（默认复用已存 `decision_latency_10node.npy`，`--remeasure-decision` 强制重测）；
  4. 写出 md/csv/tex/json。

## 数据来源（训练日志）

| 方法 | 10 节点 tb_dir | 备注 |
|------|----------------|------|
| AutoLFD (LGDRL) | `model/twitter_largescale/0530/1829/PPO_dnn` | 本机 |
| SAC | `AutoScaling_0619/.../twitter_largescale/0531/1355/SAC` | 归档机 |
| DeepScaler | `AutoScaling_0619/.../twitter_largescale/0607/1653/DeepScaler` | 归档机 |
| 20 节点对照 | `model/twitter_xlargescale/0619/2040/{PPO_dnn,SAC,DeepScaler}` | 本机 |

> 20 节点决策延迟取自 `test_output/twitter_xlargescale/latency.npy`（SAC 存在 `RL Agent` key 下）。

## 关键结论（twitter_largescale，2× RTX 4080）

- AutoLFD：~10000 iterations，SPS≈1.4k，单步决策 ~1.7 ms（mean），参数量 ~66 万（2.5 MB）。
- 完整数值见 `training_overhead_10node.md`。

## 回应审稿意见

回应 **R2-5**：补充了 AutoLFD 与 baseline（SAC / DeepScaler）的训练 wall time、收敛 epoch、SPS、模型大小、单步决策延迟的定量对比，说明 AutoLFD 训练可承受、推理延迟对边缘在线部署可接受。
