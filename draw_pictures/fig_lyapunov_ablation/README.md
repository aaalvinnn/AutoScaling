# Fig.Lyapunov — Lyapunov 消融（成本控制 / 预算平稳）

> 对应二次意见 **#2（第一组意见 3）**。审稿人要求"设计一个奖励不含 Lyapunov 项、改用延迟+成本加权求和的变体"来量化 Lyapunov 的贡献；师兄补充：要体现出 Lyapunov 对**成本的控制**和**长期预算的平稳**，**单独出折线图给审稿人看**（不放正文）。

## 场景

- 数据集：`twitter_largescale`（`C=35`, `y_weight=0.2`, `y_weight_train=0.1`）
- 每个 agent 跑 1 个 episode（288 个 timeslot，`seed=1037`），记录逐时隙 delay 与 cost。

## 生成文件

| 文件 | 内容 |
|------|------|
| `Delay-lyapunov-ablation-twitter.pdf/png` | Full vs w/o Lyapunov 的逐时隙延迟 |
| `Cost-lyapunov-ablation-twitter.pdf/png` | Full vs w/o Lyapunov 的逐时隙开销 |
| `data/*.npy` | 两个变体的 `t_all` / `cost` 原始数组（rollout 阶段自动生成） |

## 绘图脚本

- `draw.py` — 加载 2 个 PPO 模型，各自跑完整 rollout，画 delay/cost 折线。

## 变体

| 变体 | 论文标签 | 配置 flag | 说明 |
|------|----------|-----------|------|
| AutoLFD (Full) | AutoLFD (Full) | 无 | 含 Lyapunov drift-plus-penalty |
| w/o Lyapunov | w/o Lyapunov | `ablation_no_lyapunov_strict=True` | 奖励改为 `-(y_w·cost + delay)`，去 Lyapunov 项 |

## 配色

| 变体 | 颜色 | 线型 |
|------|------|------|
| AutoLFD (Full) | `#c22f2f`（红） | `-`（实线） |
| w/o Lyapunov | `#8E6EC8`（紫） | `--`（虚线） |

## 模型路径

| 变体 | 模型路径 |
|------|----------|
| AutoLFD (Full) | `model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth`（canonical，另一台机器训练） |
| w/o Lyapunov | `model/twitter_largescale_no_lyapunov_strict/0616/0124/PPO_dnn/model_dnn_5000.pth`（本机 `as` 环境，epoch 5000；best 几乎相同） |

## 当前结果（实测均值）

| 变体 | Mean Delay | Mean Cost |
|------|------------|-----------|
| AutoLFD (Full) | 7.86 | 30.93 |
| w/o Lyapunov | 7.20 | 28.90 |

## ⚠️ 已知问题（待解决）

当前 strict 变体在 delay/cost 上**都略优于 Full**，与"Lyapunov 有贡献"的预期相反。根因是**两个模型训练条件不一致**：
- Full = canonical（0530/1829，另一台机器、训练超参未知）
- w/o Lyapunov = 本机重训（`as` 环境，`y_weight_train=0.1`）

不是单变量对照。本机重训的 Full（0616/1027、0617/1110）又稳定退化到 `cost≈8 / delay 11~19` 的过度省成本最优。**需要同条件重训一对 Full + strict** 才能得到干净的 Lyapunov 消融结论。

## 相关代码改动

- `env/environment.py` `_cal_route_delay`：修复"请求链中某微服务被缩到 0 实例"时的广播崩溃（返回 `T_max`，与 `_cal_execution_delay` 一致），full/strict 不触发。
- `train_ablation.py`：支持 `python train_ablation.py <variant> <config>` 指定数据集（默认 alibaba）；device 读 `AUTOSCALING_DEVICE` 环境变量（默认 `cuda:1`）。
