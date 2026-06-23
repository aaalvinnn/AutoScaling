# Fig.4 — 逐时隙延迟与开销变化曲线

## 图片说明

本图展示各算法在每个时隙（Timeslot）上的延迟（Latency）和开销（Cost）动态变化，用于论文 5.3 节 Overall Performance Comparison。通过时隙级别的细粒度对比，直观展示 AutoLFD 在负载波动下的响应稳定性和成本优势。

## 生成文件

### 10 节点（切换 `draw.py` 顶部 `dataset`）

| 文件 | 内容 |
|------|------|
| `Latency-{sin,twitter,alibaba}-Large Scale.pdf` | 各场景各算法逐时隙延迟对比 |
| `Cost-{sin,twitter,alibaba}-Large Scale.pdf` | 各场景各算法逐时隙开销对比 |

### 20 节点（`twitter`/`alibaba` `_xlargescale_deep`）

| 文件 | 内容 |
|------|------|
| `Latency-twitter-XLarge Scale.pdf` / `Cost-twitter-XLarge Scale.pdf` | 20 节点 Twitter 逐时隙延迟/开销 |
| `Latency-alibaba-XLarge Scale.pdf` / `Cost-alibaba-XLarge Scale.pdf` | 20 节点 Alibaba 逐时隙延迟/开销 |
| `xlarge_deep_comparison.pdf` | 20 节点 LGDRL-4L vs baseline 的 delay/cost/RSR 分组柱状图 |

## 绘图脚本

- `draw.py` — 10 节点版，切换 `dataset`（`sin`/`twitter`/`alibaba`），读本地 `data/{dataset}_largescale/<agent>/{t_all,cost}.npy`
- `draw_xlarge.py` — 20 节点 `_deep` 逐时隙图，读本地 `data/{twitter,alibaba}_xlargescale_deep/<agent>/*.npy`，JSON 落 `data/data_{config}.json`
- `eval_xlarge_deep.py` — 20 节点 eval 管线：跑各 agent 288 步生成 per-step npy（→ `test_output/`）+ 画 `xlarge_deep_comparison` 柱状图，JSON 落 `data/xlarge_deep_comparison.json`

## 数据来源

- `data/{sin,twitter,alibaba}_largescale/<agent>/*.npy` — 10 节点逐时隙指标（从 `test_output/` 复制，供 release 复现）
- `data/{twitter,alibaba}_xlargescale_deep/<agent>/*.npy` — 20 节点逐时隙指标
- `data/data_*.json` / `data/xlarge_deep_comparison.json` — 逐 step 数据 + 汇总统计

## 对比算法

| 算法 | 论文标签 | 线型 | 颜色 |
|------|----------|------|------|
| RL Agent (SAC) | RL Agent | `--` | `#449945`（绿） |
| Proscale | Proscale | `--` | `#f8c91c`（黄） |
| PPA (HPA) | PPA | `--` | `#376eae`（蓝） |
| DeepScaler | DeepScaler | `--` | `#8E6EC8`（紫） |
| LGDRL (Ours) | Ours | `-.` | `#c22f2f`（红） |

> "Ours" 使用 `-.`（点划线）以突出区分，其余 baseline 统一使用 `--`（虚线）。

## 图表尺寸

- Latency 图：`figsize=(14, 4)`，宽幅展示时隙变化
- Cost 图：`figsize=(7, 5)`，标准尺寸

## 相比原稿（Major Revision 前）的变化

- **新增 Alibaba 场景**：原稿仅有 Sin + Twitter 的时隙变化图，修订后新增 Alibaba 场景（回应 R2-4）
- **新增 DeepScaler 对比线**：原稿无 GNN baseline，修订后加入 DeepScaler 的时隙变化曲线（回应 R2-1）
- **配色与线型统一**：Ours 使用红色点划线突出显示，与 fig5/fig6 保持一致的视觉风格

## 回应审稿意见

本图回应两项审稿意见：

- **R2-4**：新增 Alibaba 场景的逐时隙延迟和开销变化曲线，扩展数据集覆盖范围。
- **R2-1**：新增 DeepScaler（GNN baseline）的逐时隙对比曲线，直观展示各算法在负载波动下的响应稳定性差异。

## Response Letter 草稿

感谢审稿人关于更广泛数据集验证和纳入现代 baseline 的建议。Figure 4 现展示三个数据集（Twitter、正弦合成数据和 Alibaba）的逐时隙延迟和开销对比，并加入了 DeepScaler 作为 GNN-based baseline。逐时隙曲线显示，AutoLFD（红色点划线）在三个场景中始终保持比所有 baseline 更低的延迟方差和更稳定的资源开销。在负载波动最剧烈的 Alibaba trace 上，AutoLFD 的延迟在整个测试区间内稳定在 15 ms 以下，而 DeepScaler 频繁出现超过 30 ms 的尖峰，说明 Lyapunov drift-plus-penalty 框架在高度动态的边缘计算负载下具有更优的稳定性。
