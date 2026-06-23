# Fig.3 — 训练收敛曲线

## 图片说明

本图展示 AutoLFD（PPO）在三种负载场景下的训练收敛过程，以 epoch-reward 曲线呈现，用于论文 5.2 节 Algorithm Convergence。展示算法在不同负载模式下的收敛速度和稳定性。

## 生成文件

| 文件 | 内容 |
|------|------|
| `Train.pdf` | 10 节点三场景训练收敛合并图（Twitter + Sin + Alibaba） |
| `Convergence-twitter-xlarge.pdf` | 20 节点 Twitter 三方法（LGDRL/DeepScaler/SAC）收敛对比 |
| `Convergence-twitter-xlarge-deep.pdf` | 20 节点 LGDRL「4 层 + 20000 epoch」容量实验收敛曲线 |

## 绘图脚本

- `draw.py` — 10 节点三场景收敛图（读 `data/0603/*.json` + alibaba 的 TensorBoard）
- `draw_xlarge_methods.py` — 20 节点 Twitter 三方法收敛对比（纯 Python 解析 tfevents），JSON 落 `data/data_convergence_twitter_xlargescale.json`
- `draw_xlarge_twitter_deep.py` — 20 节点 LGDRL 4 层长训收敛，JSON 落 `data/data_convergence_twitter_xlarge_deep.json`

## 数据来源

| 场景 | 数据来源 | 说明 |
|------|----------|------|
| Twitter | `data/0603/Twitter.json` | 历史 JSON 格式训练日志 |
| Sin | `data/0603/Sin-1.json` + `Sin-2.json` | 两段拼接（共 20000 epoch） |
| Alibaba | `model/alibaba_largescale/0602/1440/PPO_dnn/` | TensorBoard EventAccumulator 读取 |

## 配色方案

- Twitter: `#82B0D2`（浅蓝）
- Sin: `#8ECFC9`（浅绿）
- Alibaba: `#FA7F6F`（浅橙红）

## 其他文件

- `data/0603/*.json` — 10 节点 Twitter/Sin 训练日志（历史 JSON 格式）
- `data/data_convergence_twitter_xlarge{,scale,deep}*.json` — 20 节点收敛曲线逐 epoch 数据（release 复现）
- `DeepScaler_*.png` — DeepScaler baseline 的收敛曲线调试图（非论文用图）

## 数据来源与复现

- **10 节点 Train.pdf**：Twitter/Sin 读 `data/0603/*.json`；Alibaba 读 `model/alibaba_largescale/0602/1440/PPO_dnn` 的 TensorBoard。
- **20 节点三方法**（`draw_xlarge_methods.py`）：读 `model/twitter_xlargescale/0619/2040/{PPO_dnn,DeepScaler,SAC}` 的 `charts/y`。
- **20 节点 LGDRL 4 层长训**（`draw_xlarge_twitter_deep.py`）：读 `model/twitter_xlargescale_deep/0621/1617/PPO_dnn` 的 `charts/y`。
- **复现**：直接跑对应脚本（纯 Python tfevents 解析，绕开本机 tensorboard C reader 腐蚀），逐 epoch 数据落 `data/data_convergence_*.json`。

## 相比原稿（Major Revision 前）的变化

- **新增 Alibaba 收敛曲线**：原稿仅有 Sin + Twitter 的训练曲线，修订后新增 Alibaba 场景的收敛展示

## 回应审稿意见

本图回应两项审稿意见：

- **R2-5**：审稿人指出 "The paper does not discuss the training overhead or convergence time of the MDRL approach"，要求补充训练收敛和开销分析。Fig.3 直接展示三场景下的 epoch-reward 收敛曲线，直观呈现收敛速度和稳定性；配套的训练开销 Table （Section 5.2）进一步提供 wall time、GPU 显存、推理延迟等定量数据。
- **R2-4**：新增 Alibaba 场景的收敛曲线，扩展数据集覆盖范围。

## Response Letter 草稿

感谢审稿人关于训练收敛和计算开销的意见。为此，我们在 Figure 3 中新增了 Alibaba trace 的训练收敛曲线，与原有的 Twitter 和正弦 trace 合并展示。epoch-reward 曲线表明，AutoLFD 在三种负载类型上均稳定收敛，其中负载变化更复杂的 Alibaba trace 需要更多 epoch 才能收敛，这是其更高负载变异性的自然结果。针对训练开销（R2-5），我们以 twitter_largescale 场景为代表进行了详细测量：在 2× RTX 4080 环境下，使用 16 个并行环境训练 10,000 iterations，总挂钟时间约 17.6 小时，平均每秒处理 1,453 步（SPS），模型参数量为 657,948（2.51 MB）；推理阶段单步决策延迟均值 1.72 ms，中位数 1.25 ms（CPU 推理），对边缘在线部署而言完全可接受。完整数据见 training_overhead.md。
