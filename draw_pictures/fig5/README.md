# Fig.5 — 参数敏感性分析

## 图片说明

本图展示 AutoLFD 中两个关键超参数的敏感性分析，用于论文 5.5 节 Parameter Sensitivity Analysis：

1. **V 参数**（Lyapunov 权衡系数）：控制延迟-开销权衡的松紧程度，V 越大越偏向降低开销、容忍更高延迟
2. **δ 参数**（队列截断门限 $N_{\mathrm{change}}$）：控制每步最大实例增减范围，影响动作空间的探索粒度

## 生成文件

| 文件 | 内容 |
|------|------|
| `Latency-twitter-Large Scale-V.pdf` | Twitter 场景不同 V 值下的逐时隙延迟对比 |
| `Cost-twitter-Large Scale-V.pdf` | Twitter 场景不同 V 值下的逐时隙开销对比 |
| `Latency-sin-Large Scale-delta.pdf` | Sin 场景不同 δ 值下的逐时隙延迟对比 |
| `Cost-sin-Large Scale-delta.pdf` | Sin 场景不同 δ 值下的逐时隙开销对比 |

## 绘图脚本

| 脚本 | 内容 | 场景 |
|------|------|------|
| `V.py` | V 参数敏感性（V=10, 100, 1000） | Twitter Large Scale |
| `delta.py` | δ 参数敏感性（δ=1, 2, 3） | Sin Large Scale |

## V 参数实验设置

- V=10：较小的 Lyapunov 权重，更激进地优化即时性能
- V=100：默认值，平衡延迟与开销
- V=1000：较大的 Lyapunov 权重，更保守、偏向队列稳定

## δ 参数实验设置

- δ=1（$N_{\mathrm{change}}=[-1,1]$）：每步最多增减 1 个实例，动作空间小
- δ=2（$N_{\mathrm{change}}=[-2,2]$）：每步最多增减 2 个实例
- δ=3（$N_{\mathrm{change}}=[-3,3]$）：每步最多增减 3 个实例，动作空间大

## 配色方案

- 第一条线: `#449945`（绿）
- 第二条线: `#376eae`（蓝）
- 第三条线: `#c22f2f`（红）

## 其他文件

- `model_autoscaling_0426/` — 早期版本模型存档（历史参考）

## 相比原稿（Major Revision 前）的变化

- **完全新增**：原稿中没有参数敏感性分析章节和对应图表。这是 Major Revision 新增的 5.5 节内容
- **回应审稿意见**：
  - R1-1 关注动作空间可扩展性 → δ 参数实验展示了不同动作范围的影响
  - R1-3 关注 Lyapunov 优化的贡献 → V 参数实验展示了不同 Lyapunov 权重的效果差异
- **当前覆盖范围**：V 参数在 Twitter 场景完成，δ 参数在 Sin 场景完成；Alibaba 场景的参数敏感性实验尚未完成

## 回应审稿意见

本图回应两项审稿意见：

- **R1-1**：审稿人关注 "The flat action space (server, microservice, change_amount) may not scale well to larger clusters"。δ 参数敏感性分析展示了不同动作粒度（δ=1,2,3）下的性能一致性，说明尽管使用了扁平动作空间，在一定范围内动作空间大小的变化对性能影响有限；论文 5.5 节进一步讨论了层次化动作空间作为未来扩展方向。
- **R1-3**：审稿人要求 "Adding an ablation study to illustrate a direct quantification of the specific contribution of Lyapunov optimization"。V 参数敏感性分析展示了不同 Lyapunov 权衡系数（V=10,100,1000）下延迟与开销的 tradeoff 变化，验证了 Lyapunov 框架提供了可调节的延迟-开销平衡机制。

## Response Letter 草稿

感谢审稿人关于（1）扁平动作空间的可扩展性和（2）量化 Lyapunov 优化具体贡献的意见。针对动作空间可扩展性问题（R1-1），我们对动作粒度参数 δ ∈ {1, 2, 3} 进行了敏感性分析，该参数控制每步实例调整的范围。如 Figure 5(a) 所示，不同 δ 值下的性能保持一致，说明 AutoLFD 的扁平动作空间在合理的粒度范围内是鲁棒的；我们在 Section 5.5 中还讨论了针对更大集群规模的层次化动作空间扩展方向。针对 Lyapunov 优化贡献的量化（R1-3），我们分析了权衡系数 V ∈ {10, 100, 1000} 的影响。Figure 5(b) 表明，较大的 V 值以更高延迟为代价获得更低的资源开销，而较小的 V 值优先降低延迟——这证实了 Lyapunov 框架为延迟-开销权衡提供了有原则且可调节的机制。
