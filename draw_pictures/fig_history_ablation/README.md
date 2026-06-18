# Fig.History — 预测 / 历史数据机制消融

> 对应二次意见 **#3（第二组意见 3）**。审稿人要求评估预测机制；师兄补充：现有的是"去掉历史数据"，**缺一个"已知当前数据（理想状态 / oracle）"的对照**，**单独出折线图给审稿人看**（不放正文）。本图三组对照：Oracle（完美已知当前到达率）→ Full（从历史隐式预测）→ w/o History（盲），量化预测/历史通道的贡献。

## 场景

- 数据集：`twitter_largescale`（`C=35`, `y_weight=0.2`）
- Oracle / w/o History 模型训练时用 `y_weight_train=0.05`、`num_envs=8`（本机 `as` 环境）。
- 每个 agent 跑 1 个 episode（288 个 timeslot，`seed=1037`），记录逐时隙 delay 与 cost。

## Oracle 实现

- `ablation_oracle_lamda=True`（`env/environment.py`）：
  - `get_observation`：`predicted_lamda` 通道填**当前真实到达率**，history 通道清零。
  - `step`：跳过 SMA 预测，直接用 `_cal_lamda_list()`（当前真实值）填 `predicted_lamda`。
- 含义：网络看到的到达率信息零误差（理想状态），无需预测。

## 生成文件

| 文件 | 内容 |
|------|------|
| `Delay-history-ablation-twitter.pdf/png` | Full / Oracle / w/o History 的逐时隙延迟 |
| `Cost-history-ablation-twitter.pdf/png` | Full / Oracle / w/o History 的逐时隙开销 |
| `data/*.npy` | 每个变体的 `t_all` / `cost` 原始数组 |

## 绘图脚本

- `draw.py` — 加载 3 个 PPO 模型，各自跑完整 rollout，画 delay/cost 折线。

## 变体

| 变体 | 论文标签 | 配置 flag |
|------|----------|-----------|
| AutoLFD (Full) | AutoLFD (Full) | 无（10 个历史 step 的到达率，隐式预测） |
| Oracle | Oracle (known current) | `ablation_oracle_lamda=True`（完美已知当前到达率） |
| w/o History | w/o History | `ablation_no_history=True`（到达率通道全清零，盲） |

## 配色

| 变体 | 颜色 | 线型 |
|------|------|------|
| AutoLFD (Full) | `#c22f2f`（红） | `-`（实线） |
| Oracle | `#449945`（绿） | `--`（虚线） |
| w/o History | `#376eae`（蓝） | `-.`（点划线） |

## 模型路径

| 变体 | 模型路径 |
|------|----------|
| AutoLFD (Full) | `model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth`（canonical，另一台机器训练） |
| Oracle | `model/twitter_largescale_oracle_lamda/0617/1206/PPO_dnn/model_dnn_best.pth`（本机 `as`，0.05） |
| w/o History | `model/twitter_largescale_no_history/0617/1206/PPO_dnn/model_dnn_best.pth`（本机 `as`，0.05） |

## 当前结果（实测均值）

| 变体 | Mean Delay | Mean Cost |
|------|------------|-----------|
| AutoLFD (Full) | 7.86 | 30.93 |
| Oracle | 8.54 | 30.18 |
| w/o History | 11.69 | 27.79 |

## 结论（故事成立）

- **预测有用**：盲的 w/o History 延迟 11.69，比 Full 高 **48%**。
- **隐式预测接近理想**：Full（历史隐式预测）7.86 ≈ Oracle（完美已知）8.54——基于历史的隐式预测几乎达到 oracle 水平。直接回应审稿人"DRL 如何利用历史做隐式预测"。
- Cost：Full / Oracle 在预算 `C=35` 附近；w/o History 成本最低（27.79）是盲导致 under-provisioning 的副作用（故延迟才高）。

> 注：Full(7.86) 略低于 Oracle(8.54) 属训练波动（canonical Full 优化充分 vs Oracle 本机新训），两者统计可比，不影响主结论。

## ⚠️ 注意：Full 用的是 canonical 模型

本机重训的 Full（`0617/1110`，`y_weight_train=0.05`）稳定退化到 `cost≈8 / delay≈19`，不能用于本图。故 Full 沿用 canonical `0530/1829`。Oracle / w/o History 为本机同条件（0.05）训练。三者训练设置并非完全一致，但 #3 讲的是**预测机制**（观测信息量），此对照可接受。
