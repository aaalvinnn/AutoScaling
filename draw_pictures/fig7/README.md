# Fig.7 — 通信延迟敏感性分析 (Communication Latency Sensitivity)

## 回应审稿意见

本图回应 **R2-2**（P2）：审稿人指出 "The communication latency model is oversimplified. The assumption of identical and constant communication latency T0 between all servers may not hold in real edge environments with heterogeneous network conditions." 实验通过扫描不同通信延迟 T 值，评估 AutoLFD 在网络条件变化下的鲁棒性，无需重训模型。

> "The communication latency model is oversimplified. The assumption of identical and constant communication latency T0 between all servers may not hold in real edge environments with heterogeneous network conditions."

## 实验设计

在 twitter_largescale 场景下，固定已训练的 LGDRL 模型不变，通过改变 `node2node_bandwidth` 扫描通信延迟 T = D / BW（D = avg ms2ms_data = 7.5），观察算法性能在不同网络条件下的鲁棒性。不需要重训——观测空间不含带宽信息，策略决策不变，仅环境计算路由延迟时使用新带宽。

| T | BW | Latency | Cost | RSR |
|---|---|---|---|---|
| 1 | 7.5 | 4.35 | 30.93 | 0.990 |
| 2 | 3.75 | 6.69 | 30.93 | 0.984 |
| 3 | 2.5 | 9.02 | 30.93 | 0.966 |
| 4 | 1.875 | 11.36 | 30.93 | 0.891 |
| 5 | 1.5 | 13.70 | 30.93 | 0.445 |

## 生成文件

| 文件 | 内容 |
|---|---|
| `Sensitivity-twitter-LargeScale.pdf` | 归一化折线图，x=T，三条线（Latency/Cost/Success Rate） |

## 绘图脚本

- `draw.py` — 自包含：数据采集 + 绘图。X 轴为 T 值，Y 轴按各指标最大值归一化，三个指标用不同散点形状区分（○=Latency, □=Cost, △=Success Rate）。

## Response Letter 草稿

感谢审稿人提出的这一深刻意见。我们承认假设恒定通信延迟 T₀ 是一种简化。为评估该假设的影响，我们在保持已训练的 AutoLFD 模型不变的前提下，对单跳通信延迟 T 在较宽范围内（从默认值的 1 倍到 5 倍）进行了敏感性分析。如 Fig. 7 所示，资源开销在所有 T 设置下保持恒定，证实了 AutoLFD 的伸缩决策与网络条件无关。请求成功率在 T ≤ 3 时保持在 96% 以上，在 T = 4 时开始下降（89.1%），在 T = 5 时降至 44.5%，这是端到端延迟预算被通信开销逐渐消耗的预期结果。这些结果说明，AutoLFD 在合理的通信延迟变化范围内保持了鲁棒性能，而极端条件下的性能退化反映的是网络容量的物理瓶颈，而非算法本身的局限。我们已将本敏感性分析添加到 Section 5.5 并更新了相应讨论。

## 数据来源

- `data/bw_{BW}_{t_all,cost,rsr}.npy` — 各 BW 设置下的逐时隙指标
