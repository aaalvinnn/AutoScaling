# AutoLFD Major Revision 修订与实验计划

> **截止日期**：2026-07-02
> **最后更新**：2026-06-10
> **目标**：把论文复现实验、Major Revision 新增实验、审稿意见回应整理成一份可执行 TODO。
> **硬件**：2 x RTX 4080 16GB
> **环境**：conda env `tcc`，Python 3.10，PyTorch 2.5.1，Gymnasium 1.0.0

---

## 0. 审稿意见与任务映射

| 编号 | 审稿意见 | 优先级 | 对应任务/产出 | 当前状态 |
|------|----------|--------|---------------|----------|
| R1-1 | 扁平动作空间 `(server, microservice, change_amount)` 在大规模集群下可能不可扩展 | P1 | 大规模实验；动作空间复杂度分析 | ❌ 未开始 |
| R1-2 | 需要说明 DRL 如何利用历史请求数据 `R_his` 做隐式预测 | P2 | 消融「w/o Historical Data」已量化 + 文字解释 | ✅ 消融完成（alibaba），📝 文字未写 |
| R1-3 | 需要消融实验量化 Lyapunov 优化贡献 | P0 | AutoLFD vs w/o Lyapunov | ✅ 消融完成（alibaba），审稿未要求多数据集 |
| R2-1 | 缺少 GNN/MARL 等前沿方法对比 | P1 | DeepScaler/GNN baseline | ✅ 训练完成，🔄 需性能对比图 |
| R2-2 | 固定通信延迟 `T0` 过于简化 | P2 | 可变通信模型或敏感性分析 | ✅ fig7 已完成（twitter_largescale，T=1~5 敏感性分析） |
| R2-3 | 预测机制未充分评估 | P0 | 消融实验「w/o Historical Data」 | ✅ 同 R1-2 |
| R2-4 | 数据集单一，只有 synthetic + Twitter | P1 | Alibaba trace 验证 | ✅ 训练+测试+图表完成 |
| R2-5 | 未讨论 MDRL 训练开销和收敛时间 | P0 | 训练/推理开销表 | 🔄 LGDRL 开销已提取（twitter_largescale，见 fig3/training_overhead.md），缺 SAC/DeepScaler 对比 |
| R2-6 | MFFD + MDRL 解耦可能导致次优 | P2 | 论文讨论；消融 w/o FFD 已量化 | ✅ 消融完成（alibaba），📝 文字未写 |

优先级说明：

- **P0**：必须完成，直接回应最关键实验质疑。
- **P1**：强烈建议完成，显著增强 Major Revision 说服力。
- **P2**：可用分析、讨论、小规模敏感性实验回应；若实现稳定再扩展。

---

## 1. 当前状态

### 1.1 已有代码与数据

**核心方法** (`methods/`)：
- ✅ PPO_dnn.py — AutoLFD/LGDRL 主算法，CNN+DNN dual-input encoder
- ✅ SAC.py — RL_agent baseline（双 Q 网络）
- ✅ FFD.py — MFFD 初始部署策略
- ✅ HPA.py, ProScaling.py, GDCScaling.py — 传统 autoscaling baseline
- ✅ NoScaling.py, RandomScaling.py — 消融参考 baseline
- ✅ Predicter.py — 到达率预测模块
- ✅ DeepScaler.py — GNN baseline（**已训练**）

**环境** (`env/`)：
- ✅ `environment.py` — Gymnasium env，含 M/M/c 排队模型、Lyapunov 优化、通信延迟
- ✅ `datastruct.py` — 状态/动作/图数据结构

**配置** (`env/configs/`)：
- ✅ `config_twitter_{small, middle, large}scale.py`（3 个）
- ✅ `config_sin_{small, middle, large}scale.py`（3 个）
- ✅ `config_twitter_largescale_requests15.py`（requests=15 变体）
- ✅ `config_alibaba_largescale.py`

**数据** (`data/`)：
- ✅ `loads-sin.txt` — 正弦负载 + 随机扰动
- ✅ `loads-twitter.txt` — Twitter trace 2022-11-01
- ✅ `loads-alibaba-v2022.txt` — Alibaba Cluster Trace v2022

### 1.2 环境确认

- ✅ Gymnasium seed patch 已应用（`reset_async` 统一并行种子）
- ✅ conda env `tcc` 可用（Python 3.10, PyTorch 2.5.1）
- ✅ 2 × RTX 4080 16GB 可用

### 1.3 已完成训练模型

#### PPO (LGDRL) — 主算法

| 配置 | 目录 | 状态 |
|------|------|------|
| `twitter_largescale` | `model/twitter_largescale/0530/1829/PPO_dnn/` | ✅ 20000ep 完成，含 best checkpoint + TensorBoard |
| `sin_largescale` | `model/sin_largescale/0529/2157/PPO_dnn/` | ✅ 20000ep 完成，含 best checkpoint |
| `sin_largescale` | `model/sin_largescale/0529/2145/PPO_dnn/` | ✅ 20000ep 完成（第二次运行） |
| `sin_largescale` | `model/sin_largescale/0531/1359/PPO_dnn/` | ✅ 20000ep 完成（第三次运行） |
| `alibaba_largescale` | `model/alibaba_largescale/0602/1440/PPO_dnn/` | ✅ 20000ep 完成，含 best checkpoint |
| `alibaba_largescale` | `model/alibaba_largescale/0602/1427/PPO_dnn/` | ✅ 另一次运行 |
| `twitter_largescale` (V=2) | `model/twitter_largescale/0605/V=2/PPO_dnn/` | ✅ 20000ep 完成 |
| `twitter_largescale` (V=1000) | `model/twitter_largescale/0605/V=1000/PPO_dnn/` | ✅ 20000ep 完成 |
| `twitter_largescale` (V=0.3) | `trained_models/twitter_largescale/0412_V0_3/` | ✅ 20000ep 完成 |
| `config_twitter_largescale_requests15` | `model/config_twitter_largescale_requests15/0524/1210/PPO_dnn/` | ✅ 完成 |
| `twitter_smallscale` | `trained_models/twitter_smallscale/0312/` | ✅ 历史版本 |
| `twitter_middlescale` | `trained_models/twitter_middlescale/0315/` | ✅ 历史版本 |
| `sin_smallscale` | `trained_models/sin_smallscale/0314/` | ✅ 历史版本 |
| `sin_middlescale` | `trained_models/sin_middlescale/0312/` | ✅ 历史版本 |

#### SAC (RL_agent) — DRL baseline

| 配置 | 目录 | 状态 |
|------|------|------|
| `twitter_largescale` | `model/twitter_largescale/*/SAC/`（0531-0603 共 8 次运行） | ✅ 多次训练完成 |
| `sin_largescale` | `model/sin_largescale/0529/2158/SAC/`, `0529/2146/SAC/` | ✅ 完成 |
| `alibaba_largescale` | `model/alibaba_largescale/0602/1429/SAC/`, `0602/1446/SAC/` | ✅ 完成（2 次） |
| `sin_largescale` | `trained_models/sin_largescale/0314/1637/SAC/` | ✅ 历史版本 |
| `twitter_largescale` | `trained_models/twitter_largescale/0313/1046/SAC/` | ✅ 历史版本 |
| `twitter_smallscale` | `model/twitter_smallscale/*/SAC/`（0304-0311 共 8 次运行） | ✅ 多次训练完成 |

#### DeepScaler (GNN) — GNN baseline

> **Reference:** DeepScaler: Holistic Autoscaling for Microservices Based on Spatiotemporal GNN with Adaptive Graph Learning. ASE 2023. DOI: 10.1109/ASE56229.2023.00038

| 配置 | 目录 | 状态 | 备注 |
|------|------|------|------|
| `alibaba_largescale` | `model/alibaba_largescale/0604/2126/DeepScaler/` | ✅ 训练完成 | 旧版 6 层 DNN (1.72M params) |
| `alibaba_largescale` | `model/alibaba_largescale/0605/1412/DeepScaler/` | ✅ 训练完成 | 旧版 6 层 DNN (1.72M params) |
| `alibaba_largescale` | `model/alibaba_largescale/0606/1702/DeepScaler/` | ✅ 训练完成 | 2 层 DNN (0.67M params)，加权求和 reward，RSR 仅 0.58 |
| `sin_largescale` | `model/sin_largescale/0607/1653/DeepScaler/` | ✅ 训练完成 | 3 层 DNN (1.16M params)，加权求和 reward |
| `twitter_largescale` | `model/twitter_largescale/0607/1653/DeepScaler/` | ✅ 训练完成 | 3 层 DNN (1.16M params)，加权求和 reward |
| `alibaba_largescale` | `model/alibaba_largescale/0607/1653/DeepScaler/` | ✅ 训练完成 | 3 层 DNN (1.16M params)，加权求和 reward |

##### DeepScaler v0607/1653 详细测试结果（3 层 DNN, 加权求和 reward, dnn_hidden=512）

**total_reward：**

| Agent | sin | twitter | alibaba |
|---|---|---|---|
| LGDRL | -2683.95 | -2412.38 | -2389.37 |
| DeepScaler | -2792.39 | -4662.20 | -3862.42 |

**fig2 逐指标：**

| 数据集 | Agent | Latency | Cost | RSR |
|---|---|---|---|---|
| sin | DeepScaler | 9.70 | **30.28** | **0.999** |
| sin | LGDRL | **9.32** | 34.22 | 0.966 |
| twitter | DeepScaler | 8.93 | **25.73** | **0.999** |
| twitter | LGDRL | **8.38** | 31.03 | 0.959 |
| alibaba | DeepScaler | 13.41 | 38.55 | 0.576 |
| alibaba | LGDRL | **8.30** | **36.50** | **0.958** |

**问题：** sin/twitter 上 DeepScaler 的 Cost 和 RSR 反优于 LGDRL（Cost 低、RSR 高），原因是加权求和 reward 不含 Qt 排队积压惩罚，DeepScaler 学到了"少部署实例"策略。alibaba 因负载波动大导致 RSR 崩溃。

**下一步：** 将 DeepScaler reward 改为 Lyapunov（与 LGDRL 一致的 `-y`），重训 sin/twitter。如果效果差可恢复此版本。

#### 消融变体（仅 alibaba_largescale）

| 变体 | 目录 | 状态 |
|------|------|------|
| w/o Lyapunov | `model/alibaba_largescale_no_lyapunov/0603/2043/PPO_dnn/` | ✅ 训练完成 |
| w/o Lyapunov (strict) | `model/alibaba_largescale_no_lyapunov_strict/0605/1413/PPO_dnn/` | ✅ 训练完成 |
| w/o Historical Data | `model/alibaba_largescale_no_history/0603/2043/PPO_dnn/` | ✅ 训练完成 |
| w/o FFD | `model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn/` | ✅ 训练完成 |

#### CNN 模型（历史）

| 配置 | 目录 | 状态 |
|------|------|------|
| `twitter_largescale` | `model/0107/184309/` | ✅ model_cnn_best 保存 |

### 1.4 V 参数与 δ 参数实验进展

#### V 参数实验（Lyapunov 权衡系数）

| V 值 | 目录 | 说明 |
|------|------|------|
| V=0.1, 0.2, 0.3 | `trained_models/params_exp/V/0_{1,2,3}/` | ✅ 已完成 + 已测试 |
| V=1 | `model/twitter_largescale/0605/V=1/PPO_dnn/` | ✅ 已训练 |
| V=2 | `model/twitter_largescale/0605/V=2/PPO_dnn/` | ✅ 已训练 + 已测试 |
| V=10 | `draw_pictures/fig5/model_autoscaling_0426/model/delta=1/0419/` 等 | ✅ 已训练（旧版模型） |
| V=100 (default) | 多个版本 | ✅ 已训练 + 已测试 |
| V=1000 | `model/twitter_largescale/0605/V=1000/PPO_dnn/` | ✅ 已训练 + 已测试 |
| N_change 变体 | `trained_models/params_exp/V/N_change=1时/`, `N_change=3/` | ✅ 已完成 |

#### δ 参数实验（队列截断门限）

| δ 值 | 目录 | 说明 |
|------|------|------|
| δ=1 | `trained_models/params_exp/delta/1/` + `0610/delta=1/` | ✅ 多版本训练 + 测试 |
| δ=2 | `trained_models/params_exp/delta/2/` + `0610/delta=2/` | ✅ 多版本训练 + 测试 |
| δ=3 | `trained_models/params_exp/delta/3/` | ✅ 多版本训练 + 测试 |

### 1.5 已有测试/实验结果

#### twitter_largescale（较完整）

`test_output/twitter_largescale/` 包含：
- ✅ **Baseline 对比数据** (`data/`): LGDRL, HPA, Proscale — 各 19 个指标 (.npy)
- ✅ **V 参数测试数据**: V=0.1~1000 多版本覆盖
- ✅ **δ 参数测试数据**: δ=1, 2, 3
- ✅ **可视化图表**: 26 张 PNG
- ❌ 缺失：NoScaling, Random, RL_agent (SAC) baseline 测试数据

#### sin_largescale

`test_output/sin_largescale/` 包含：
- ✅ **Baseline 对比数据** (`data/`): LGDRL, HPA, HPA_ideal, Proscale — 各 19 个指标
- ✅ **可视化图表**: 26 张 PNG
- ❌ 缺失：NoScaling, Random, RL_agent (SAC)

#### alibaba_largescale

`test_output/alibaba_largescale/` 包含：
- ✅ 测试输出目录存在

#### twitter_middlescale / sin_middlescale / twitter_smallscale / sin_smallscale

- ✅ 全部有测试输出（图表 + data 目录），包含 LGDRL/HPA/Proscale 等

#### config_twitter_largescale_requests15

- ✅ 完整测试输出（24 张图表 + data 目录）

### 1.6 已有图表 (`draw_pictures/`)

| 图表 | 路径 | 内容 | 状态 |
|------|------|------|------|
| Fig.1 负载曲线 | `fig1/` | `loads-sin.pdf`, `loads-twitter.pdf`, **`loads-alibaba.pdf`** | ✅ 三场景完成 |
| Fig.2 收敛/结果 | `fig2/` | `sin-Large Scale.pdf`, `twitter-Large Scale.pdf`, **`alibaba-Large Scale.pdf`** | ✅ 三场景完成 |
| Fig.3 训练曲线 | `fig3/` | `Train.pdf`（PPO+SAC, Sin+Twitter），含 `data/` JSON | ✅ 完成 |
| Fig.4 时隙变化 | `fig4/` | 延迟/成本时隙图（**sin + twitter + alibaba** Large Scale） | ✅ 三场景完成 |
| Fig.5 参数敏感性 | `fig5/` | V 参数（Cost/Latency vs V），δ 参数（`delta.py`） | ✅ 完成（仅 sin/twitter） |
| **Fig.6 消融实验** | `fig6/` | **消融对比图（alibaba）**: Latency/Cost/SuccessRate，5 变体 | ✅ **新增完成** |
| **Fig.7 通信延迟敏感性** | `fig7/` | **通信延迟敏感性（twitter_largescale）**: T=1~5 扫描 | ✅ **新增完成** |
| 数据导出 | `export_excel.py` | 批量导出实验数据到 Excel | ✅ |
| README 标准化 | `fig1-7/README.md` | 各图 README 含审稿意见映射 + 中文 response letter 草稿 | ✅ 完成 |

### 1.7 消融策略说明

消融通过 config 中的 flag 控制（`ablation_no_lyapunov_strict`, `ablation_no_history`, `ablation_no_ffd`），无需 CLI 参数：

- **w/o Lyapunov**：`ablation_no_lyapunov_strict=True`，奖励改为加权和
- **w/o Historical Data**：`ablation_no_history=True`，移除 state 历史通道（ch 3-5）
- **w/o FFD**：`ablation_no_ffd=True`，随机初始部署
- ✅ Alibaba 场景三个消融变体均已训练完成
- ❌ Sin/Twitter 场景消融变体尚未训练

### 1.8 已完成 vs 待做

#### ✅ 已完成

- [x] Sin + Twitter 三规格（small/middle/large）PPO 训练
- [x] Sin + Twitter large-scale SAC 训练（RL_agent baseline）
- [x] **Alibaba large-scale PPO 训练**（20000ep）
- [x] **Alibaba large-scale SAC 训练**（2 次）
- [x] **Alibaba config 创建**（`config_alibaba_largescale.py`）
- [x] **Alibaba 测试输出**
- [x] **消融实验（alibaba）**: w/o Lyapunov, w/o History, w/o FFD — 全部训练完成
- [x] **DeepScaler GNN 训练**（alibaba，2 次）
- [x] **Fig.1 负载曲线**：Sin + Twitter + Alibaba 三合一
- [x] **Fig.2 收敛/结果**：三场景 Large Scale
- [x] **Fig.3 训练曲线**：PPO+SAC, Sin+Twitter
- [x] **Fig.4 时隙变化**：三场景 Latency + Cost
- [x] **Fig.5 参数敏感性**：V 参数 + δ 参数
- [x] **Fig.6 消融对比**：alibaba 场景 5 变体（Latency/Cost/SuccessRate）
- [x] **Fig.7 通信延迟敏感性**：twitter_largescale T=1~5 扫描
- [x] **训练开销统计（LGDRL）**：twitter_largescale 挂钟时间/推理延迟（见 fig3/training_overhead.md）
- [x] **各图 README 标准化**：fig1-7 均含审稿意见映射 + 中文 response letter 草稿
- [x] Twitter large-scale V 参数实验（V=0.1~1000，多版本覆盖）
- [x] Twitter large-scale δ 参数实验（δ=1,2,3）
- [x] N_change 变体实验
- [x] Twitter/Sin large-scale baseline 测试（LGDRL vs HPA/Proscale）
- [x] Twitter/Sin 所有规格测试输出与可视化图表
- [x] Alibaba Cluster Trace v2022 数据下载与预处理（`data/loads-alibaba-v2022.txt`）
- [x] Predicter 模块实现

---

#### ⬜ P0 — 必须完成

| 章节 | 任务 | 回应的审稿意见 | 预计工作量 | 当前进度 |
|------|------|---------------|-----------|----------|
| 5.2 | **训练开销统计**（wall time / GPU / inference） | R2-5 | 中 | 🔄 LGDRL twitter_largescale 已完成（见 fig3/training_overhead.md），缺 SAC/DeepScaler 对比 |
| 5.4 | **消融结果汇总表 (Table)** — 仅 alibaba 场景 | R1-3 | 低 | ❌ fig6 有图但缺论文级数值表（均值±std） |
| 5.3 | **NoScaling/Random/SAC baseline 补测**（twitter + sin + alibaba） | — | 低 | ❌ 三个场景均缺 |
| 5.3 | **Alibaba 全 baseline 对比图**（HPA/ProScaling/NoScaling/Random/SAC/LGDRL） | R2-4 | 低 | ❌ 缺除 LGDRL 外的 baseline 对比 |

#### ⬜ P1 — 强烈建议

| 章节 | 任务 | 回应的审稿意见 | 预计工作量 | 当前进度 |
|------|------|---------------|-----------|----------|
| 5.3 | **DeepScaler 性能对比图** | R2-1 | 低 | 🔄 模型已训练，需测试 + 绘图 |
| 5.4 | **消融实验补 sin/twitter**（可选，增强说服力） | R1-3 | 高（6 次训练 × ~4h） | ❌ 时间允许再补 |
| 5.5.1 | **V 参数图统一**（已有 V=0.1~1000，需补 alibaba 场景 + 统一绘图） | — | 低 | 🔄 数据大部分有 |
| 5.5.1 | **δ 参数图补全**（缺 twitter/alibaba 的 δ 实验） | — | 中 | 🔄 仅 sin 有 |
| 5.5.2 | **大规模可扩展性**（30/50+ nodes） | R1-1 | 高 | ❌ 未开始，无 xlargescale config |

#### ⬜ P2 — 分析/文字回应

| 章节 | 任务 | 回应的审稿意见 | 当前进度 |
|------|------|---------------|----------|
| 5.5.3 | 通信延迟敏感性（测试阶段 T0 变化） | R2-2 | ✅ fig7 已完成（twitter_largescale，T=1~5） |
| 7.2 | 历史数据利用文字分析 | R1-2 | 🔄 消融数据已有（alibaba），需撰写文字 |
| 7.3 | MFFD+MDRL 解耦讨论 | R2-6 | 🔄 消融数据已有（alibaba w/o FFD），需撰写文字 |

> **已决策不单独做**：预测精度（MAE/RMSE）→ 合并至消融"w/o Historical Data"（消融数据已齐）

---

## 2. 总体执行路线

```text
Phase 0：最小复现验证          ✅ 已完成
  -> Phase 1：复现原论文图表     ✅ 已完成（含 Alibaba 扩展）
  -> Phase 2：P0 Major Revision  🔄 进行中（消融 alibaba ✅，剩余：开销统计 + baseline 补测）
  -> Phase 3：P1 扩展实验        ⬜ 部分进行中（DeepScaler 已训练，缺对比图）
  -> Phase 4：P2 分析与论文回应  🔄 部分完成（R2-2 fig7 ✅，各图 README response letter 草稿已写）
```

---

## 实验大纲（论文章节映射）

### 实验总览

| 章节 | 对应原论文 | 审稿意见 | 优先级 | 状态 | 产出 |
|------|-----------|----------|--------|------|------|
| 5.1 Experiment Settings | 5.1（更新） | — | — | ✅ 完成 | Alibaba 场景 + 参数表 + Fig.1 负载曲线（三合一） |
| 5.2 Algorithm Convergence & Overhead | 5.2（Fig.6） | R2-5 | **P0** | 🔄 LGDRL 完成，缺 SAC/DeepScaler | Fig.3 收敛曲线 ✅ + training_overhead.md（LGDRL twitter） 🔄 |
| 5.3 Overall Performance | 5.2（Fig.7-9） | R2-4 | **P0** | 🔄 部分完成 | Fig.2/4 三场景 ✅，缺全 baseline 对比 |
| 5.4 Ablation Study | **新增** | R1-2, R1-3, R2-3 | **P0** | ✅ alibaba 完成 | Fig.6 alibaba ✅ + Table 需整理 |
| 5.5 Parameter Sensitivity Analysis | 5.2（Fig.10）**新增** | R1-1, R2-2（✅ fig7 完成） | **P1** | 🔄 部分完成 | Fig.5 V/δ ✅（sin/twitter），Fig.7 通信延迟 ✅，缺 alibaba |

> **注**：R1-2（历史数据利用）和 R2-3（预测机制）统一通过消融实验中的「w/o Historical Data」变体回应——Alibaba 消融已完成，直接量化了历史到达率通道的贡献。

### 5.1 Experiment Settings（更新）

**状态**：✅ 完成

已完成：
- ✅ `config_alibaba_largescale.py` 创建
- ✅ Alibaba trace 数据预处理（`data/loads-alibaba-v2022.txt`）
- ✅ Fig.1 负载曲线（Sin + Twitter + Alibaba 三合一）

待做（文字）：
- [ ] 更新论文 5.1 节场景描述，新增 Alibaba trace 说明
- [ ] 更新参数表 Table I

### 5.2 Algorithm Convergence & Training Overhead（训练收敛与开销）⭐ P0

**状态**：🔄 图已完成，开销统计未做

**5.2.1 训练收敛**：✅ 已完成
- ✅ Fig.3 训练收敛曲线（PPO+SAC, Sin+Twitter）
- ✅ TensorBoard 日志齐全（三场景 PPO + SAC）

**5.2.2 训练开销统计**（回应 R2-5）：❌ 未开始

| 指标 | 数据来源 | 说明 |
|------|----------|------|
| 总训练 wall time | TensorBoard / 计时器 | 从 start 到 convergence |
| 收敛所需 epochs | reward 曲线拐点 | 以 reward 稳定为收敛标准 |
| GPU 显存峰值 | `torch.cuda.max_memory_allocated()` | 训练阶段 |
| 单步推理延迟 | action selection 计时 | 不含环境 step |
| 测试 rollout 时间 | 完整 288-slot rollout | 含环境交互 |

**实现任务**：
- [ ] `methods/PPO_dnn.py`：添加计时和显存统计
- [ ] `methods/SAC.py`：同样统计
- [ ] `methods/DeepScaler.py`：同样统计
- [ ] 输出保存到 `outputs/results/{config}/{experiment}/overhead.json`

**产出**：Table 训练开销表

| Config | Algo | Wall Time (h) | Converge Epoch | GPU Peak (GB) | Inference (ms) | Rollout (s) |
|--------|------|--------------|----------------|---------------|-----------------|-------------|
| twitter_largescale | PPO | TBD | TBD | TBD | TBD | TBD |
| twitter_largescale | SAC | TBD | TBD | TBD | TBD | TBD |
| sin_largescale | PPO | TBD | TBD | TBD | TBD | TBD |
| alibaba_largescale | PPO | TBD | TBD | TBD | TBD | TBD |
| alibaba_largescale | SAC | TBD | TBD | TBD | TBD | TBD |
| alibaba_largescale | DeepScaler | TBD | TBD | TBD | TBD | TBD |

### 5.3 Overall Performance Comparison（整体性能对比）⭐ P0

**状态**：🔄 部分完成（三场景图表有，但缺全 baseline 对比）

**已完成**：
- ✅ Fig.2 收敛/结果柱状图（sin/twitter/alibaba Large Scale）
- ✅ Fig.4 时隙变化（sin/twitter/alibaba Latency + Cost）
- ✅ `test_output/alibaba_largescale/` 存在

**缺失**：
- ❌ NoScaling/Random/SAC baseline 补测（twitter + sin + alibaba）
- ❌ Alibaba 全 baseline 对比（HPA/ProScaling/NoScaling/Random/SAC vs LGDRL）
- ❌ DeepScaler 性能对比（模型已训练，缺测试 + 绘图）
- ❌ 跨数据集性能汇总 Table

**产出**：
- Fig.2 三指标柱状图（3 场景 × 6+ agents）— 需补完整 baseline
- Fig.4 延迟/成本时隙变化（3 场景）— ✅ 已完成
- Table 跨数据集性能汇总（Sin / Twitter / Alibaba 三列对比）

### 5.4 Ablation Study（消融实验）⭐ P0

**状态**：✅ Alibaba 完成（审稿未要求多数据集消融）

> **决策**：两位审稿人的消融相关意见（R1-2, R1-3, R2-3）均未要求多数据集消融。R1-3 原文仅要求 "adding an ablation study to illustrate a direct quantification of the specific contribution of Lyapunov optimization"，未提及多场景。消融在 Alibaba 数据集上完成即可，论文中写 "We conduct the ablation study on the Alibaba trace" 即可满足审稿要求。主实验性能对比已覆盖 Sin/Twitter/Alibaba 三场景（R2-4），两者互补。

**Alibaba 消融**（✅ 已完成）：
- ✅ 训练：Full, w/o Lyapunov (strict), w/o History, w/o FFD, Full+no-FFD（5 变体）
- ✅ 图表：`draw_pictures/fig6/` — Latency/Cost/SuccessRate 三张图
- ✅ 数据：`draw_pictures/fig6/data/` — 15 个 .npy 文件

**待做**：
- [ ] 从 fig6 数据中提取消融结果汇总 Table（均值±std，4 变体 × 3 指标）
- [ ] 论文 5.4 节撰写消融分析文字

**可选扩展**（P1，时间允许）：
- 如果有余力可补 sin/twitter 消融训练增强说服力，但非必须
- 参考 `draw_pictures/fig6/draw.py` 中的 `make_config(ablation_no_lyapunov_strict=True)` 方式实现

**产出**：
- Fig.6 消融对比图（alibaba）✅
- Table 消融结果汇总表（alibaba × 4 变体）❌ 需整理

### 5.5 Parameter Sensitivity Analysis（参数敏感性分析）⭐ P1

**状态**：🔄 部分完成

**5.5.1 V Parameter（Lyapunov 权衡系数）**：✅ Twitter 数据齐全
- ✅ V=0.1~1000 全覆盖（twitter_largescale）
- ✅ Fig.5 V 参数图已绘制
- ❌ 缺 alibaba 场景的 V 参数实验

**5.5.2 δ Parameter（队列截断门限）**：🔄 仅 sin
- ✅ δ=1,2,3 已训练 + 测试（sin_largescale）
- ✅ Fig.5 δ 参数图已绘制
- ❌ 缺 twitter/alibaba 的 δ 实验

**5.5.3 Scalability（集群规模可扩展性）**：❌ 未开始
- ❌ 无 xlargescale config
- ❌ 无 30/50+ nodes 实验

| 配置 | Nodes | MS | 动作空间规模 | 预计显存 | 预计训练时间 |
|------|-------|-----|-------------|----------|-------------|
| twitter_largescale | 10 | 10 | $10\times10\times7=700$ | ~4 GB | 4-6 h |
| twitter_xlargescale | 30 | 15 | $30\times15\times7=3150$ | ~8 GB | 12-18 h |
| twitter_xxlargescale | 50 | 20 | $50\times20\times7=7000$ | ~14 GB | 24-36 h |

**实施策略**：先 30-node 探针 → 稳定后 50+ nodes；若 OOM 或稀疏则报告 boundary + 讨论 hierarchical action space。

**5.5.4 Communication Latency（通信延迟敏感性）**：❌ 未开始
- 保持已训练模型不变，测试阶段变化 $T_0\in\{1, 2, 5, 10, 20\}$ ms

---

### 论文新增图表清单

| 图表编号 | 内容 | 对应章节 | 优先级 | 状态 |
|----------|------|----------|--------|------|
| Fig.1 (更新) | Sin + Twitter + Alibaba 负载曲线 | 5.1 | — | ✅ |
| Fig.2 (更新) | 三场景收敛/结果柱状图 | 5.3 | — | 🔄 需补全 baseline |
| Fig.3 (更新) | PPO + SAC 训练收敛曲线 | 5.2 | — | ✅ |
| Fig.4 (更新) | 三场景延迟/成本时隙变化 | 5.3 | — | ✅ |
| Fig.5 (更新) | V 参数 + δ 参数敏感性 | 5.5.1 | — | 🔄 缺 alibaba |
| **Fig.6 (新增)** | 消融对比图（Latency/Cost/SuccessRate, alibaba） | 5.4 | P0 | ✅ alibaba 完成 |
| Fig.7 (新增) | 通信延迟敏感性（twitter_largescale, T=1~5） | 5.5.3 | P2 | ✅ 完成 |
| **Table (新增)** | 训练开销表 | 5.2 | P0 | 🔄 LGDRL twitter 完成，缺 SAC/DeepScaler |
| **Table (新增)** | 跨数据集性能汇总 | 5.3 | P0 | ❌ |
| **Table (新增)** | 消融结果汇总（alibaba × 4 变体） | 5.4 | P0 | 🔄 alibaba 数据有，需整理成表 |
| **Table (新增)** | 可扩展性汇总 | 5.5.2 | P1 | ❌ |

---

## 3. 剩余工作量估算

### P0 任务（必须完成）

| 任务 | GPU 时间 | 人工时间 | 并行策略 |
|------|---------|---------|---------|
| 训练开销统计（加计时重跑） | 2-4h（选代表性场景） | 中（改代码） | 改完代码后单次

**P0 总 GPU 时间**：约 2-4h
**P0 总人工时间**：约 2-3 天

### P1 任务（强烈建议）

| 任务 | GPU 时间 | 人工时间 |
|------|---------|---------|
| DeepScaler 测试 + 对比图 | 0（模型已有） | 低 |
| 消融 sin/twitter（可选增强） | 12-24h | 低 |
| V/δ 参数图补全（alibaba） | 可能需要训练 | 中 |
| 大规模可扩展性（30/50 nodes） | 12-36h | 高（新 config + 训练） |

### P2 任务（文字回应）

| 任务 | 时间 |
|------|------|
| 通信延迟敏感性 | 中（实现 + 小规模测试） |
| 历史数据利用文字分析 | 低（消融数据已有） |
| MFFD+MDRL 解耦讨论 | 低（消融数据已有） |

---

## 4. 建议执行顺序

```
Week 1（06/06 - 06/12）：P0 收尾 + P1 启动
  Day 1: PPO/SAC/DeepScaler 添加计时器 → 重跑一次训练
  Day 2-3: 全 baseline 补测（NoScaling/Random/SAC/HPA/ProScaling，三场景）
  Day 3: 整理消融/性能汇总 Table
  Day 4-5: DeepScaler 测试 + 对比图

Week 2（06/13 - 06/19）：P1 扩展 + P2 文字
  Day 1-2: 大规模可扩展性实验（30 nodes 探针）或 消融补 sin/twitter（可选）
  Day 3: V/δ 参数图补全
  Day 4: 通信延迟敏感性（如果时间允许）
  Day 5: 历史数据利用 + MFFD解耦 文字分析

Week 3-4（06/20 - 07/02）：论文修改 + Response Letter
  论文正文修改
  Response letter 逐条对应
  图表/表格最终整理
```

---

## 5. 风险清单

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 大规模实验 OOM | R1-1 难完整回应 | 先跑 30 nodes，必要时报告 scalability boundary |
| 计时器改造引入 bug | 开销数据不准 | 改完后对比 reward 曲线确认训练无变化 |
| Gymnasium seed patch 缺失 | 结果不可复现 | 每次 batch run 前确认 patch |
| 时间不足跳过 P1 | 论文说服力下降 | P0 完成后评估，P1 可降级为简短讨论 |
| 通信模型改动影响所有指标 | 重跑成本高 | 优先做 sensitivity，不轻易改主实验环境 |

---

## 6. 输出目录约定

```text
model/
  {config_name}/{MMDD}/{HHMM}/{algo}/         # 训练输出
  {config_name}_{ablation}/{MMDD}/{HHMM}/{algo}/  # 消融变体

test_output/
  {config_name}/
    data/{agent}/{metric}.npy
    *.png

draw_pictures/
  fig1/  # 负载曲线
  fig2/  # 收敛/结果柱状图
  fig3/  # 训练曲线
  fig4/  # 时隙变化
  fig5/  # 参数敏感性
  fig6/  # 消融对比
  fig7/  # 通信延迟敏感性
```

---

## 7. 最终交付物

- [ ] 修订论文正文。
- [ ] Response letter：逐条对应所有 reviewer comments。
- [ ] 原论文图表重生成。
- [x] Fig.1 负载曲线（三场景）。
- [x] Fig.3 训练收敛曲线。
- [x] Fig.4 时隙变化（三场景）。
- [x] Fig.5 参数敏感性（V + δ）。
- [x] Fig.6 消融对比（alibaba）。
- [x] Fig.7 通信延迟敏感性（twitter_largescale, T=1~5）。
- [ ] Fig.2 全 baseline 对比柱状图（三场景，补全 agents）。
- [ ] Table 训练开销表（R2-5，🔄 LGDRL twitter 已完成）。
- [ ] Table 跨数据集性能汇总（Sin/Twitter/Alibaba）。
- [ ] Table 消融结果汇总（alibaba × 4 变体）。
- [ ] P1 图表/表格，如果完成：
  - [ ] DeepScaler 对比图。
  - [ ] 大规模可扩展性。
  - [x] 通信延迟敏感性（fig7）。
- [ ] 所有论文结果对应的可复现实验命令日志。
