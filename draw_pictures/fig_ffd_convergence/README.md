# Fig.FFD-Convergence — 部署质量对收敛的影响（MFFD vs 随机初始部署）

> 对应二次意见 **#6**（部署/路由解耦）。审稿人担心「MFFD 初始部署 + DRL 微调」解耦后是否次优；本图从**收敛过程**角度回应：MFFD 提供的高质量初始部署不仅影响最终性能，还**显著加速训练收敛**。

## 图片说明

对比 Full（MFFD 初始部署）与 w/o FFD（随机初始部署，重训）两条 PPO 训练的 epoch–reward 收敛曲线。reward = −charts/y（Lyapunov drift，越低越好 → 取负后越高越好）。MFFD 收敛更快、末值更高 = 部署质量帮助收敛。

## 生成文件

| 文件 | 内容 |
|------|------|
| `Convergence-FFD-ablation-alibaba.pdf/png` | alibaba_largescale 两方法收敛对比 |
| `Convergence-FFD-ablation-twitter.pdf/png` | twitter_largescale 两方法收敛对比 |

## 绘图脚本

- `draw.py` — 纯 Python 手写 tfevents 解析器读 `charts/y`（绕开本机 tensorboard C reader 在 numpy 2.2 下的腐蚀），平滑后画两条收敛曲线。顶部 `DATASET` 开关切 `alibaba`/`twitter`。

## 数据来源

- `data/data_convergence_ffd_ablation_{alibaba,twitter}.json` — 逐 epoch reward + 收敛统计（末 10% 均值、首达总提升 90% 的 epoch），由 `draw.py` 从 TensorBoard 日志抽取后落盘。

## 对照训练（TensorBoard 日志路径）

| 数据集 | Full（MFFD） | w/o FFD（random init, retrain） |
|--------|--------------|---------------------------------|
| alibaba | `model/alibaba_largescale/0602/1440/PPO_dnn` | `model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn` |
| twitter | `model/twitter_largescale/0530/1829/PPO_dnn` | `model/twitter_largescale_no_ffd/0622/1143/PPO_dnn` |

## 配色

- AutoLFD (MFFD init)：`#c22f2f`（红，实线）
- w/o FFD (random init)：`#449945`（绿，虚线）

## 回应审稿意见

回应 **#6**：MFFD 不仅给 DRL 微调一个热启动起点（见 fig_ffd_earlyslots 早时隙图），还让训练更快收敛到更好策略——MFFD 与 MDRL 是协同而非可替代关系。
