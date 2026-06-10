# Fig.6 — 消融实验（Ablation Study）

## 图片说明

本图展示 AutoLFD 在 Alibaba 大规模场景下的消融实验结果，用于论文 5.4 节 Ablation Study。通过逐一移除框架中的关键组件，量化各模块（Lyapunov 优化、历史数据通道、MFFD 初始部署）对整体性能的贡献。

## 生成文件

### 逐时隙折线图

| 文件 | 内容 |
|------|------|
| `Latency-ablation-alibaba.pdf/png` | 5 个变体的逐时隙延迟对比 |
| `Cost-ablation-alibaba.pdf/png` | 5 个变体的逐时隙开销对比 |
| `SuccessRate-ablation-alibaba.pdf/png` | 5 个变体的逐时隙成功率对比 |

### 均值柱状图

| 文件 | 内容 |
|------|------|
| `Bar-ablation-alibaba.pdf/png` | 归一化分组柱状图：x 轴为 3 个指标（Latency/Cost/Success Rate），每组 5 根柱子对应 5 个消融变体，值按该指标最大值归一化 |

### 历史文件（已弃用）

| 文件 | 说明 |
|------|------|
| `Bar-Cost-ablation-alibaba.pdf/png` | 旧版独立柱状图（已合并） |
| `Bar-Latency-ablation-alibaba.pdf/png` | 旧版独立柱状图（已合并） |
| `Bar-SuccessRate-ablation-alibaba.pdf/png` | 旧版独立柱状图（已合并） |

## 绘图脚本

- `draw.py` — 加载 5 个消融变体的模型，运行完整 rollout 收集数据，绘制折线图和柱状图

## 消融变体

| 变体 | 论文标签 | 配置 flag | 说明 |
|------|----------|-----------|------|
| Full AutoLFD | AutoLFD (Full) | 无 | 完整框架 |
| w/o Lyapunov | w/o Lyapunov | `ablation_no_lyapunov_strict=True` | 奖励改为加权求和，移除 Lyapunov drift-plus-penalty |
| w/o History | w/o Historical Data | `ablation_no_history=True` | 移除观测中的历史到达率通道（ch 3-5） |
| w/o FFD (retrained) | w/o FFD (retrained) | `ablation_no_ffd=True` | 随机初始部署替代 MFFD，模型重新训练 |
| Full + no-FFD | Full + no-FFD | `ablation_no_ffd=True` | 使用 Full 模型在随机部署环境下测试（不重训），衡量部署质量对推理的影响 |

## 配色方案

| 变体 | 颜色 | 线型 |
|------|------|------|
| Full AutoLFD | `#8E6EC8`（紫） | `-.`（点划线） |
| w/o Lyapunov | `#c22f2f`（红） | `--`（虚线） |
| w/o History | `#376eae`（蓝） | `--`（虚线） |
| w/o FFD | `#449945`（绿） | `--`（虚线） |
| Full + no-FFD | `#f8c91c`（黄） | `--`（虚线） |

## 数据来源

- `data/` — 15 个 `.npy` 文件（5 变体 × 3 指标），由 `draw.py` 在 rollout 阶段自动生成

## 模型路径

| 变体 | 模型路径 |
|------|----------|
| Full | `model/alibaba_largescale/0602/1440/PPO_dnn/model_dnn_best.pth` |
| w/o Lyapunov | `model/alibaba_largescale_no_lyapunov_strict/0605/1413/PPO_dnn/model_dnn_best.pth` |
| w/o History | `model/alibaba_largescale_no_history/0603/2043/PPO_dnn/model_dnn_best.pth` |
| w/o FFD | `model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn/model_dnn_best.pth` |

## 相比原稿（Major Revision 前）的变化

- **完全新增**：原稿中没有消融实验章节和对应图表。这是 Major Revision 最重要的新增内容之一
- **回应审稿意见**：
  - R1-3 明确要求 "adding an ablation study to illustrate a direct quantification of the specific contribution of Lyapunov optimization" → w/o Lyapunov 变体直接量化了 Lyapunov 优化的贡献
  - R1-2 要求说明 DRL 如何利用历史请求数据 → w/o History 变体量化了历史到达率通道的贡献
  - R2-3 要求评估预测机制 → 与 R1-2 统一通过 w/o History 回应
  - R2-6 关注 MFFD+MDRL 解耦可能导致的次优 → w/o FFD 变体量化了 MFFD 初始部署的贡献
- **仅在 Alibaba 场景完成**：审稿人未要求多数据集消融，Alibaba 场景已满足审稿要求

## 回应审稿意见

本图是 Major Revision 最核心的新增内容，直接回应四项审稿意见：

- **R1-3**（P0）：审稿人明确要求 "adding an ablation study to illustrate a direct quantification of the specific contribution of Lyapunov optimization" → w/o Lyapunov 变体量化了 Lyapunov drift-plus-penalty 框架的贡献。
- **R1-2**（P2）：审稿人要求说明 "how the DRL agent takes advantage of historical request data to perform implicit prediction" → w/o History 变体量化了历史到达率通道对成功率的贡献。
- **R2-3**（P0）：审稿人要求 "The prediction mechanism... should be evaluated more thoroughly" → 与 R1-2 统一通过 w/o History 变体回应。
- **R2-6**（P2）：审稿人关注 "The decoupling of MFFD and MDRL may lead to sub-optimal results" → w/o FFD 变体量化了 MFFD 初始部署的贡献。

## Response Letter 草稿

感谢审稿人建议通过消融实验量化各模块的具体贡献。我们在 Alibaba trace 上进行了全面的消融实验，如新增的 Figure 6 所示。主要发现如下：（1）移除 Lyapunov 优化（w/o Lyapunov，回应 R1-3）导致资源开销增加 4.9%（38.30 vs. 36.50），说明缺少 drift-plus-penalty 框架时，智能体倾向于过度配置资源以追求低延迟，牺牲了资源效率——Lyapunov 公式化对有原则的延迟-开销平衡至关重要。（2）移除历史到达率通道（w/o History，回应 R1-2 和 R2-3）使成功率从 95.8% 下降至 81.5%（相对下降 15%），延迟标准差增加 81%，表明基于历史数据的隐式负载预测对于在波动负载下维持服务质量不可或缺。（3）用随机分配替代 MFFD 初始部署并重新训练（w/o FFD retrained，回应 R2-6）可达到与完整模型相近的最终性能（成功率 95.2% vs. 95.8%），说明 DRL 微调阶段可以部分弥补次优初始部署的不足。然而，在不重新训练的情况下用完整模型在随机部署下测试（Full + no-FFD），成功率骤降至 53.0%，证实 MFFD 提供的热启动是 DRL 微调有效进行的前提条件。这些结果共同表明，AutoLFD 的各组件对整体性能均有实质性贡献，其协同组合优于任何单一消融变体。
