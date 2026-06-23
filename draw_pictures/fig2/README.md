# Fig.2 — 多算法整体性能对比柱状图

## 图片说明

本图以归一化分组柱状图的形式展示各自动伸缩算法在三个指标（Latency、Cost、Success Rate）上的整体性能对比，用于论文 5.3 节 Overall Performance Comparison。每个指标的值按该指标中所有算法的最大值做归一化，便于跨指标对比。

## 生成文件

| 文件 | 内容 |
|------|------|
| `sin-Large Scale.pdf` | Sin 大规模场景（10 节点 / 10 微服务）的 5 算法对比 |
| `twitter-Large Scale.pdf` | Twitter 大规模场景的 5 算法对比 |
| `alibaba-Large Scale.pdf` | Alibaba 大规模场景的 5 算法对比 |
| `twitter-XLarge Scale.pdf` | Twitter **20 节点**场景（`twitter_xlargescale_deep`）的 5 算法对比 |

## 绘图脚本

- `draw.py` — 10 节点版，切换顶部 `DATASET`（`sin`/`twitter`/`alibaba`），读本地 `data/{dataset}_largescale/<agent>/{t_all,cost,request_success_rate}.npy`
- `draw_xlarge_twitter.py` — Twitter 20 节点版，读本地 `data/twitter_xlargescale_deep/<agent>/*.npy`（原始 per-step npy 由 `fig4/eval_xlarge_deep.py` 生成）
- `draw_old.py` — 原稿版本（保留备查），仅支持 Sin/Twitter，4 个 agent，3 张独立子图

## 数据来源

- `data/{sin,twitter,alibaba}_largescale/<agent>/*.npy` — 10 节点三数据集各 agent 逐时隙指标（从 `test_output/` 复制，供 release 复现）
- `data/twitter_xlargescale_deep/<agent>/*.npy` — 20 节点 Twitter 各 agent 指标

## 对比算法

| 算法 | 论文标签 | 类型 |
|------|----------|------|
| PPA (HPA) | PPA | 传统启发式 baseline |
| Proscale | Proscale | 比例缩放 baseline |
| RL Agent (SAC) | RL Agent | DRL baseline（连续动作空间） |
| DeepScaler | DeepScaler | GNN baseline（ASE 2023） |
| LGDRL (Ours) | Ours | AutoLFD 主算法 |

## 配色方案

- PPA: `#8ECFC9`（浅绿）
- Proscale: `#FFBE7A`（浅橙）
- RL Agent: `#82B0D2`（浅蓝）
- DeepScaler: `#BEB8DC`（浅紫）
- Ours: `#FA7F6F`（浅橙红）

## 相比原稿（Major Revision 前）的变化

- **新增 Alibaba 场景**：原稿仅有 Sin + Twitter，修订后新增 Alibaba Cluster Trace 场景（回应 R2-4）
- **新增 DeepScaler baseline**：原稿仅 4 个算法（Random, RL Agent, Proscale, LGDRL），修订后新增 DeepScaler（GNN）对比（回应 R2-1："include more recent works...using graph neural networks"）

## 回应审稿意见

本图回应两项审稿意见：

- **R2-4**：审稿人要求 "Broader validation using multiple real-world microservice traces"。原稿仅涵盖 Sin 和 Twitter 两个场景，修订后新增 Alibaba 场景，Fig.2 现覆盖三个数据集的整体性能对比。
- **R2-1**：审稿人指出 "The comparison with existing works needs to be strengthened by including more recent works for microservice auto-scaling, e.g., those using graph neural networks"。修订后新增 DeepScaler 作为 baseline，形成 5 算法对比。

> **DeepScaler 出处**：DeepScaler: Holistic Autoscaling for Microservices Based on Spatiotemporal GNN with Adaptive Graph Learning. ASE 2023. DOI: 10.1109/ASE56229.2023.00038

## Response Letter 草稿

感谢审稿人关于（1）在更广泛数据集上验证框架和（2）纳入更新的基于学习的自动伸缩方法进行对比的建议。我们在 Figure 2 中将对比扩展到了三个数据集——Twitter、正弦合成数据和 Alibaba Cluster Trace v2022，并新增了 DeepScaler（ASE 2023，基于 GNN 的微服务自动伸缩方法）作为 baseline。归一化柱状图显示，AutoLFD 在三个指标（延迟、开销、成功率）和三个数据集上均取得最优或接近最优的性能。值得注意的是，在负载波动剧烈的 Alibaba trace 上，DeepScaler 的成功率大幅下降，而 AutoLFD 保持了鲁棒的性能，验证了 Lyapunov 引导的优化框架在复杂真实场景下的优势。
