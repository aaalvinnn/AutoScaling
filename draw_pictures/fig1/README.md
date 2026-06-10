# Fig.1 — 请求到达率负载曲线

## 图片说明

本图展示三种负载轨迹（Load Trace）的请求到达率随时间变化的曲线，用于论文 5.1 节 Experiment Settings，直观呈现不同数据集的负载特征差异。

## 生成文件

| 文件 | 内容 |
|------|------|
| `loads.pdf` | 三条负载曲线合并在一张图中（Twitter + Sin + Alibaba） |

## 绘图脚本

- `draw.py` — 读取 `data/` 目录下的三个负载文件，绘制合并曲线图

## 数据来源

| 数据集 | 文件 | 特征 |
|--------|------|------|
| Twitter | `data/loads-twitter.txt` | Twitter 2022-11-01 真实集群 trace，突发流量明显 |
| Sin | `data/loads-sin.txt` | 正弦波 + 随机扰动，周期性负载 |
| Alibaba | `data/loads-alibaba-v2022.txt` | Alibaba Cluster Trace v2022，大规模真实负载 |

## 配色方案

- Twitter: `#82B0D2`（浅蓝）
- Sin: `#8ECFC9`（浅绿）
- Alibaba: `#FA7F6F`（浅橙红）

## 相比原稿（Major Revision 前）的变化

- **新增 Alibaba 负载曲线**：原稿仅有 Sin + Twitter 两种负载，Major Revision 应审稿人 R2-4 要求（"Broader validation using multiple real-world microservice traces, e.g., from Alibaba Cluster Trace"）新增了 Alibaba Cluster Trace v2022
- **合并为单图**：原稿中 Sin 和 Twitter 分别绘制，修订后三种负载合并到一张图中，便于对比不同数据集的负载模式差异

## 回应审稿意见

本图主要回应 **R2-4**（"Broader validation using multiple real-world microservice traces, e.g., from Alibaba Cluster Trace"）。原稿仅使用 Twitter Trace 和合成正弦波两种负载，审稿人建议增加更多真实微服务 trace 以增强实验说服力。修订后新增 Alibaba Cluster Trace v2022，三种负载合并于 Fig.1，直观展示负载多样性和评估广度。

## Response Letter 草稿

感谢审稿人建议在更多真实负载 trace 上进行验证。我们在修订稿中引入了 Alibaba Cluster Trace v2022，这是来自全球最大云平台之一的大规模生产环境 trace。Figure 1 现已将三种数据集的请求到达率曲线合并到同一张图中——Twitter（真实社交媒体 trace）、正弦合成数据和 Alibaba（生产云 trace），便于直接对比三者截然不同的负载模式。Twitter trace 呈现尖锐的瞬时突发特征，正弦 trace 表现为平滑的周期性变化，而 Alibaba trace 则展示了持续高流量伴随不规则波动的生产级负载特性，三者共同体现了评估场景的多样性，为研究结论的泛化性提供了更强有力的支撑。
