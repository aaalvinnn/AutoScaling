# Fig.FFD-EarlySlots — 初始部署质量的早期时隙影响（意见 #6）

> 对应二次意见 **#6**（部署/路由解耦）。师兄预测：随机初始部署在前几个时隙应明显差于 MFFD，因为 DRL 还没来得及微调。本图把**前 N 个时隙放大**单独出图，直接展示初始部署质量对早期推理的影响，并补完整时隙曲线看是否恢复。

## 图片说明

三条线的逐时隙 Latency / Cost：
- **AutoLFD (MFFD init)** — 完整框架，MFFD 初始部署
- **w/o FFD (random, retrain)** — 随机初始部署 + 重训
- **Full + no-FFD (only test)** — Full 模型直接在随机部署下测试（不重训），衡量部署质量对推理的纯影响

每张图两种画幅：
- **EarlySlots**：前 40 个时隙放大（部署质量对早期推理的影响）
- **Full**：完整时隙曲线（看全貌 / 是否恢复）

## 生成文件

| 文件 | 内容 |
|------|------|
| `EarlySlots-FFD-ablation-alibaba.pdf/png` | alibaba 前 40 时隙 Latency+Cost（1×2） |
| `Full-FFD-ablation-alibaba.pdf/png` | alibaba 完整时隙 Latency+Cost（1×2） |
| `EarlySlots-{Latency,Cost}-FFD-ablation-twitter.pdf/png` | twitter 前 40 时隙单指标 |
| `Full-{Latency,Cost}-FFD-ablation-twitter.pdf/png` | twitter 完整时隙单指标 |

## 绘图脚本

| 脚本 | 数据集 | 数据来源 |
|------|--------|----------|
| `draw.py` | alibaba | 读 `fig6/data/*.npy`（fig6 rollout 产物） |
| `draw_twitter.py` | twitter | 自跑 rollout，存到本地 `data/*.npy` |

## 数据来源

- `data/*.npy` — twitter 三条线的 `t_all` / `cost` 逐时隙数组（由 `draw_twitter.py` rollout 生成）。alibaba 版复用 `draw_pictures/fig6/data/`。

## 配色

- AutoLFD (MFFD init)：`#c22f2f`（红，实线）
- w/o FFD (random, retrain)：`#449945`（绿，虚线）
- Full + no-FFD (only test)：`#f8c91c`（黄，虚线）

## 回应审稿意见

回应 **#6**：早时隙图显示随机初始部署在前若干时隙延迟/开销明显恶化（尤其 Full+no-FFD only-test，部署未微调），证实 MFFD 初始部署是 DRL 微调有效进行的前提；完整时隙曲线显示重训后 w/o FFD 能逐步追上，说明 DRL 微调可部分弥补次优初始部署——两者协同。
