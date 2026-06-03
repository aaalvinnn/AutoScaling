# AutoLFD Major Revision 修订与实验计划

> **截止日期**：2026-07-02
> **目标**：把论文复现实验、Major Revision 新增实验、审稿意见回应整理成一份可执行 TODO。
> **硬件**：2 x RTX 4080 16GB
> **环境**：conda env `tcc`，Python 3.10，PyTorch 2.5.1，Gymnasium 1.0.0

---

## 0. 审稿意见与任务映射

| 编号 | 审稿意见 | 优先级 | 对应任务/产出 |
|------|----------|--------|---------------|
| R1-1 | 扁平动作空间 `(server, microservice, change_amount)` 在大规模集群下可能不可扩展 | P1 | 大规模实验；动作空间复杂度分析 |
| R1-2 | 需要说明 DRL 如何利用历史请求数据 `R_his` 做隐式预测 | P2 | 历史特征消融/可视化；论文解释 |
| R1-3 | 需要消融实验量化 Lyapunov 优化贡献 | P0 | AutoLFD vs w/o Lyapunov |
| R2-1 | 缺少 GNN/MARL 等前沿方法对比 | P1 | DeepScaler/GNN baseline |
| R2-2 | 固定通信延迟 `T0` 过于简化 | P2 | 可变通信模型或敏感性分析 |
| R2-3 | 预测机制未充分评估 | P0 | 消融实验「w/o Historical Data」（见 5.4），不再单设预测精度实验 |
| R2-4 | 数据集单一，只有 synthetic + Twitter | P1 | Alibaba/Google trace 验证 |
| R2-5 | 未讨论 MDRL 训练开销和收敛时间 | P0 | 训练/推理开销表 |
| R2-6 | MFFD + MDRL 解耦可能导致次优 | P2 | 论文讨论；可选端到端/GNN 对比 |

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
- ✅ DeepScaler.py — GNN baseline 代码骨架（**未训练**）

**环境** (`env/`)：
- ✅ `environment.py` — Gymnasium env，含 M/M/c 排队模型、Lyapunov 优化、通信延迟
- ✅ `datastruct.py` — 状态/动作/图数据结构

**配置** (`env/configs/`)：
- ✅ `config_twitter_{small, middle, large}scale.py`（3 个）
- ✅ `config_sin_{small, middle, large}scale.py`（3 个）
- ✅ `config_twitter_largescale_requests15.py`（requests=15 变体）

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
| `sin_largescale` | `trained_models/sin_largescale/0314/1637/SAC/` | ✅ 历史版本 |
| `twitter_largescale` | `trained_models/twitter_largescale/0313/1046/SAC/` | ✅ 历史版本 |
| `twitter_smallscale` | `model/twitter_smallscale/*/SAC/`（0304-0311 共 8 次运行） | ✅ 多次训练完成 |

#### CNN 模型（历史）

| 配置 | 目录 | 状态 |
|------|------|------|
| `twitter_largescale` | `model/0107/184309/` | ✅ model_cnn_best 保存 |

### 1.4 V 参数与 δ 参数实验进展

#### V 参数实验（Lyapunov 权衡系数）

| V 值 | 目录 | 说明 |
|------|------|------|
| V=1 | `model/twitter_largescale/0605/V=1/PPO_dnn/`（或 trained_models/params_exp/V/0604/V=1） | ✅ 已训练 |
| V=2 | `model/twitter_largescale/0605/V=2/PPO_dnn/` | ✅ 已训练 + 已测试 |
| V=10 | `draw_pictures/fig5/model_autoscaling_0426/model/delta=1/0419/` 等 | ✅ 已训练（旧版模型） |
| V=100 (default) | 多个版本 | ✅ 已训练 + 已测试 |
| V=1000 | `model/twitter_largescale/0605/V=1000/PPO_dnn/` | ✅ 已训练 + 已测试 |
| V=0.1, 0.2, 0.3 | `trained_models/params_exp/V/0_{1,2,3}/` | ✅ 已完成 + 已测试 |
| N_change 变体 | `trained_models/params_exp/V/N_change=1时/`, `N_change=3/` | ✅ 已完成 |

#### δ 参数实验（队列截断门限）

| δ 值 | 目录 | 说明 |
|------|------|------|
| δ=1 | `trained_models/params_exp/delta/1/` + `0610/delta=1/` | ✅ 多版本训练 + 测试 |
| δ=2 | `trained_models/params_exp/delta/2/` + `0610/delta=2/` | ✅ 多版本训练 + 测试 |
| δ=3 | `trained_models/params_exp/delta/3/` | ✅ 多版本训练 + 测试 |

### 1.5 已有测试/实验结果

#### twitter_largescale（最完整）

`test_output/twitter_largescale/` 包含：
- ✅ **Baseline 对比数据** (`data/`): LGDRL, HPA, Proscale — 各 19 个指标 (.npy)，含 `t_all`, `cost`, `request_success_rate`, `Qt`, `r` 等
- ✅ **V 参数测试数据**: V=1, 2, 10, 100, 1000（新），V=0.1, 0.2, 0.3（新），V=1, 10, 100（旧版）
- ✅ **δ 参数测试数据**: δ=1, 2, 3（含 new 和 old 版本）
- ✅ **可视化图表**: 26 张 PNG（含 cost_bar, t_all_bar, request_success_rate_bar, Qt, r, lamda, vload 等）
- ❌ 缺失：NoScaling, Random, RL_agent (SAC) baseline 测试数据

#### sin_largescale

`test_output/sin_largescale/` 包含：
- ✅ **Baseline 对比数据** (`data/`): LGDRL, HPA, HPA_ideal, Proscale — 各 19 个指标
- ✅ **可视化图表**: 26 张 PNG
- ❌ 缺失：NoScaling, Random, RL_agent (SAC)

#### twitter_middlescale / sin_middlescale / twitter_smallscale / sin_smallscale

- ✅ 全部有测试输出（图表 + data 目录），包含 LGDRL/HPA/Proscale 等

#### config_twitter_largescale_requests15

- ✅ 完整测试输出（24 张图表 + data 目录）

### 1.6 已有图表 (`draw_pictures/`)

| 图表 | 路径 | 内容 |
|------|------|------|
| Fig.1 负载曲线 | `fig1/` | `loads-sin.pdf`, `loads-twitter.pdf` |
| Fig.2 收敛/结果 | `fig2/` | `sin-Large Scale.pdf`, `twitter-Large Scale.pdf` |
| Fig.3 训练曲线 | `fig3/` | `Train.pdf`（PPO+SAC, Sin+Twitter），含 `data/` JSON |
| Fig.4 时隙变化 | `fig4/` | 延迟/成本时隙图（sin+twitter Large Scale） |
| Fig.5 参数敏感性 | `fig5/` | V 参数（Cost/Latency vs V），δ 参数（`delta.py`），含旧版模型数据 |
| 数据导出 | `export_excel.py` | 批量导出实验数据到 Excel |

### 1.7 消融策略说明

- **w/o Lyapunov**：通过 `--ablation no_lyapunov` 切换到 `weighted` reward
- **w/o Historical Data**：通过 `--ablation no_prediction` 移除 state 历史通道（ch 3-5），统一回应 R1-2 和 R2-3
- **w/o FFD**：通过 `--ablation no_ffd` 随机初始部署
- 上述三个消融均 **未开始训练**，属于 Section 5.4 的内容

### 1.8 已完成 vs 待做

#### ✅ 已完成

- [x] Sin + Twitter 三规格（small/middle/large）PPO 训练
- [x] Sin + Twitter large-scale SAC 训练（RL_agent baseline）
- [x] Twitter large-scale V 参数实验（V=0.1~1000，多版本覆盖）
- [x] Twitter large-scale δ 参数实验（δ=1,2,3）
- [x] N_change 变体实验
- [x] Twitter/Sin large-scale baseline 测试（LGDRL vs HPA/Proscale）
- [x] Twitter/Sin 所有规格测试输出与可视化图表
- [x] Fig.1-5 初版图表生成（draw_pictures/）
- [x] Alibaba Cluster Trace v2022 数据下载与预处理（`data/loads-alibaba-v2022.txt`）
- [x] Predicter 模块实现
- [x] DeepScaler GNN 代码骨架

#### 🔄 进行中

| 章节 | 任务 | 预计 | 备注 |
|------|------|------|------|
| 5.3 | Alibaba 全量训练 (20000ep) | ~8h | ⚠️ Alibaba config 尚未创建 |

#### ⬜ P0 — 必须完成

| 章节 | 任务 | 回应的审稿意见 | 预计工作量 | 当前进度 |
|------|------|---------------|-----------|----------|
| 5.1 | **Alibaba 配置创建**（`alibaba_largescale` 等） | — | 低 | ❌ config 文件未创建 |
| 5.2 | **训练收敛曲线**（PPO+SAC, 3 场景） | — | 低 | TensorBoard 已有，需绘图 |
| 5.2 | **训练开销统计**（wall time / GPU / inference） | R2-5 | 中 | ❌ 需在 PPO/SAC 加计时器 |
| 5.3 | **Alibaba 训练 + 测试** | R2-4 | 中 | ❌ 未开始 |
| 5.4 | **消融实验**: w/o Lyapunov | R1-3 | 各 20000ep ≈ 11h | ❌ 未开始 |
| 5.4 | **消融实验**: w/o Historical Data | R1-2, R2-3 | 同上 | ❌ 未开始 |
| 5.4 | **消融实验**: w/o FFD | — | 同上 | ❌ 未开始 |
| 5.3 | NoScaling/Random/RL_agent baseline 补测 | — | 低 | ❌ twitter+sin largescale 缺此数据 |

#### ⬜ P1 — 强烈建议

| 章节 | 任务 | 回应的审稿意见 | 预计工作量 | 当前进度 |
|------|------|---------------|-----------|----------|
| 5.5.1 | **V 参数实验整理**（已有 V=0.1~1000，需补 V=10 最新训练） | — | 低 | 🔄 数据大都有，需统一绘图 |
| 5.5.2 | **大规模可扩展性**（30/50+ nodes） | R1-1 | 高 | ❌ 未开始，无 xlargescale config |
| 6.3 | **DeepScaler/GNN 训练与对比** | R2-1 | 中 | ❌ 代码骨架存在，未训练 |

#### ⬜ P2 — 分析/文字回应

| 章节 | 任务 | 回应的审稿意见 | 当前进度 |
|------|------|---------------|----------|
| 5.5.3 | 通信延迟敏感性（测试阶段 T0 变化） | R2-2 | ❌ 未开始 |
| 7.2 | 历史数据利用文字分析 | R1-2 | ❌ 未开始 |
| 7.3 | MFFD+MDRL 解耦讨论 | R2-6 | ❌ 未开始 |

> **已决策不单独做**：预测精度（MAE/RMSE）→ 合并至消融"w/o Historical Data"（Section 5.2 已说明）
>
> **⚠️ Alibaba 进展修正**：TODO 之前标记 config 为 ✅，但 `env/configs/` 中实际无 alibaba 相关配置文件，需先创建 config 才能开始训练。

---

## 2. 总体执行路线

```text
Phase 0：最小复现验证
  -> Phase 1：复现原论文图表
  -> Phase 2：P0 Major Revision 新增实验
  -> Phase 3：P1 扩展实验
  -> Phase 4：P2 分析与论文回应
```

原则：先确认 Phase 0 能复现原论文趋势，再启动耗时的 P1/P2 实验，避免白跑 GPU。

---

## 实验大纲（论文章节映射）

### 实验总览

| 章节 | 对应原论文 | 审稿意见 | 优先级 | 状态 | 产出 |
|------|-----------|----------|--------|------|------|
| 5.1 Experiment Settings | 5.1（更新） | — | — | 🔄 需修改 | 新增 Alibaba/large-scale 场景 + 更新参数表 |
| 5.2 Algorithm Convergence & Overhead | 5.2（Fig.6） | R2-5 | **P0** | 🔄 需重跑 | Fig.5 收敛曲线（PPO+SAC）+ Table 训练开销 |
| 5.3 Overall Performance | 5.2（Fig.7-9） | R2-4 | **P1** | 🔄 需重跑 | Fig.6-8 性能对比 + Table 汇总（含 Alibaba） |
| 5.4 Ablation Study | **新增** | R1-2, R1-3, R2-3 | **P0** | ❌ 未开始 | Fig.9 消融对比 + Table 消融表 |
| 5.5 Parameter Sensitivity Analysis | 5.2（Fig.10）**新增** | R1-1, R2-2 | **P1** | 🔄 部分需补 | Fig.10-12 敏感性分析（V / 规模 / 通信延迟） |

> **注**：R1-2（历史数据利用）和 R2-3（预测机制）统一通过消融实验中的「w/o Historical Data」变体回应——论文未提及具体预测模型，state 中历史到达率通道即为唯一预测信息来源，消融对比直接量化其贡献。不再单设预测精度/噪声实验章节。

### 5.1 Experiment Settings（更新）

**对应原论文 5.1** | **状态**：🔄 需修改

原有场景：
- **Sin 负载**：$a=15, b=10, T_1=288$，随机扰动 $n_i\in[-5,5]$
- **Twitter 负载**：2022-11-01 Twitter trace，数学变换适配实验规模
- **节点规模**：small (5), middle (7), large (10)

Major Revision 新增：
- **Alibaba 负载**：Alibaba Cluster Trace v2022，`data/loads-alibaba-v2022.txt`
- **超大规模**：30/50+ nodes（`twitter_xlargescale`）
- **更新参数表**：新增 Alibaba 配置参数、通信延迟变体参数

**产出**：
- 更新后场景描述（新增 Alibaba trace 说明 + large-scale 规格）
- 更新参数表 Table I（补充新增配置项）
- Fig.4 负载曲线（Sin + Twitter + Alibaba 三合一）

### 5.2 Algorithm Convergence & Training Overhead（训练收敛与开销）⭐ P0

**对应原论文 Fig.6 + 审稿意见 R2-5** | **状态**：🔄 需重跑

**5.2.1 训练收敛**：
- PPO 训练收敛曲线（Sin + Twitter + Alibaba，3 条曲线）
- SAC 训练收敛曲线（同场景，作为 RL_agent baseline 的收敛证明）
- 横轴：training iterations，纵轴：episode reward

**命令**：
```bash
python run.py train --config twitter_largescale --algo ppo --device cuda:0
python run.py train --config sin_largescale --algo ppo --device cuda:1
python run.py train --config twitter_largescale --algo sac --device cuda:0
```

**5.2.2 训练开销统计**（回应 R2-5）：

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
- [ ] 输出保存到 `outputs/results/{config}/{experiment}/overhead.json`

**产出**：
- Fig.5 训练收敛曲线（含 PPO + SAC，3 场景）
- Table 训练开销表

| Config | Algo | Wall Time (h) | Converge Epoch | GPU Peak (GB) | Inference (ms) | Rollout (s) |
|--------|------|--------------|----------------|---------------|-----------------|-------------|
| twitter_largescale | PPO | TBD | TBD | TBD | TBD | TBD |
| twitter_largescale | SAC | TBD | TBD | TBD | TBD | TBD |
| sin_largescale | PPO | TBD | TBD | TBD | TBD | TBD |
| alibaba_largescale | PPO | TBD | TBD | TBD | TBD | TBD |

### 5.3 Overall Performance Comparison（整体性能对比）⭐ P1

**对应原论文 Fig.7-9 + 审稿意见 R2-4（Alibaba 泛化）** | **状态**：🔄 需重跑

**实验内容**：
- 对比 agents：`NoScaling`, `Random`, `PPA`, `ProScale`, `RL_agent`(SAC), `LGDRL`(PPO)
- 指标：average request latency, cost, request success rate
- 场景：Sin + Twitter（复用）+ **Alibaba（新增，回应 R2-4）**

**Alibaba 数据进展**（R2-4）：
- ✅ `data/loads-alibaba-v2022.txt` — Alibaba Cluster Trace v2022
- ✅ `configs/alibaba_smallscale.yaml`, `alibaba_middlescale.yaml`, `alibaba_largescale.yaml`
- ✅ `scripts/prepare_alibaba_v2022.py`
- [ ] 确认预处理方法，在论文/附录说明数据来源和预处理流程（文字回应）

**训练 + 测试命令**：
```bash
# 训练 Alibaba 场景（若未训练）
python run.py train --config alibaba_largescale --algo ppo --device cuda:0

# 测试 Twitter
python run.py test --config twitter_largescale \
    --model RL_agent=outputs/models/twitter_largescale/sac/best,LGDRL=outputs/models/twitter_largescale/ppo/best \
    --agents NoScaling,Random,PPA,ProScale,RL_agent,LGDRL \
    --output outputs/results/twitter_largescale/baseline

# 测试 Alibaba
python run.py test --config alibaba_largescale \
    --model RL_agent=outputs/models/alibaba_largescale/sac/best,LGDRL=outputs/models/alibaba_largescale/ppo/best \
    --agents NoScaling,Random,PPA,ProScale,RL_agent,LGDRL \
    --output outputs/results/alibaba_largescale/baseline
```

**产出**：
- Fig.6 三指标柱状图（3 场景 × 6 agents）
- Fig.7 延迟时隙变化（3 场景，LGDRL vs baselines）
- Fig.8 成本时隙变化（3 场景，LGDRL vs baselines）
- Table 跨数据集性能汇总（Sin / Twitter / Alibaba 三列对比）

### 5.4 Ablation Study（消融实验）⭐ P0

**对应审稿意见 R1-2（历史数据利用）, R1-3（Lyapunov 贡献）, R2-3（预测机制）** | **状态**：❌ 未开始 | **优先级 P0**

**目标**：量化 AutoLFD 各组件的独立贡献。

**实验变体**：

| 变体 | 描述 | 消融目标 | 对应审稿意见 |
|------|------|----------|-------------|
| **AutoLFD** (Full) | 完整三阶段框架 | 基准 | — |
| **w/o Lyapunov** | 奖励函数改为加权和 $-(\alpha\cdot cost + \beta\cdot mean\_delay)$ | Lyapunov drift-plus-penalty 的贡献 | R1-3 |
| **w/o Historical Data** | state 移除历史到达率通道（ch 3-5），DRL 仅靠当前部署和资源状态决策 | 历史到达率信息的贡献 | R1-2, R2-3 |
| **w/o FFD** | 随机/中性初始部署替代 MFFD | 初始部署策略的贡献 | — |

> **说明**：论文中状态定义仅提及 $R_{his}$（历史到达率），未描述具体的预测模型。代码中 state 通道 3-5 统一承载历史到达率信息（包含 SMA 平滑值与原始历史序列）。消融时通过 `--ablation no_prediction` 移除全部三个通道，对比「有历史到达率」vs「无历史到达率」的性能差异。此消融直接回应 R1-2（DRL 如何利用 $R_{his}$）和 R2-3（预测信息的贡献），无需单独设置预测精度实验章节。

**训练命令**：
```bash
python run.py train --config twitter_largescale --algo ppo --ablation no_lyapunov --device cuda:0
python run.py train --config sin_largescale --algo ppo --ablation no_lyapunov --device cuda:1
python run.py train --config twitter_largescale --algo ppo --ablation no_prediction --device cuda:0
python run.py train --config twitter_largescale --algo ppo --ablation no_ffd --device cuda:0
```

**产出**：
- Fig.9 消融对比图（延迟/成本/成功率，4 变体对比）
- Table 消融结果汇总表

| Variant | Latency (ms) | Cost | Success Rate | vs Full (Δ) |
|---------|-------------|------|-------------|-------------|
| AutoLFD (Full) | TBD | TBD | TBD | — |
| w/o Lyapunov | TBD | TBD | TBD | TBD |
| w/o Historical Data | TBD | TBD | TBD | TBD |
| w/o FFD | TBD | TBD | TBD | TBD |

### 5.5 Parameter Sensitivity Analysis（参数敏感性分析）⭐ P1

**对应原论文 Fig.10 + 审稿意见 R1-1, R2-2** | **状态**：🔄 部分需补 | **优先级 P1**

统一章节分析三类参数对系统性能的敏感性。

---

**5.5.1 V Parameter（Lyapunov 权衡系数）**

**对应原论文 Fig.10** | **状态**：🔄 需补跑 V=10/1000

- 固定其他参数，变化 $V\in\{10, 100, 1000\}$
- 观察虚拟队列 $Q(t)$ 稳定性、延迟-成本 trade-off

```bash
for V in 10 1000; do
    python run.py train --config twitter_largescale --algo ppo --v "$V" --device cuda:0
done
```

> 确保模型保存路径区分 V 值，避免覆盖 `best` checkpoint。

**产出**：Fig.10 V 参数影响（延迟 + 成本 vs V）

---

**5.5.2 Scalability（集群规模可扩展性）**

**对应审稿意见 R1-1** | **状态**：❌ 未开始

**目标**：验证扁平动作空间 $(s_i, m_j, N_{change})$ 在 30/50+ nodes 下的训练可行性。

| 配置 | Nodes | MS | 动作空间规模 | 预计显存 | 预计训练时间 |
|------|-------|-----|-------------|----------|-------------|
| twitter_largescale | 10 | 10 | $10\times10\times7=700$ | ~4 GB | 4-6 h |
| twitter_xlargescale | 30 | 15 | $30\times15\times7=3150$ | ~8 GB | 12-18 h |
| twitter_xxlargescale | 50 | 20 | $50\times20\times7=7000$ | ~14 GB | 24-36 h |

**实施策略**：先 30-node 探针 → 稳定后 50+ nodes；若 OOM 或稀疏则报告 boundary + 讨论 hierarchical action space。

**新增配置**：`configs/twitter_xlargescale.yaml`, `configs/twitter_xxlargescale.yaml`

**产出**：Fig.11 大规模性能 + 训练效率 vs 集群规模

---

**5.5.3 Communication Latency（通信延迟敏感性）**

**对应审稿意见 R2-2** | **状态**：❌ 未开始

**方案**（保守 — 不重跑主实验）：保持已训练模型不变，测试阶段变化 $T_0\in\{1, 2, 5, 10, 20\}$ ms 或注入随机扰动。

**实现任务**：
- [ ] `configs/base.yaml`：增加 `T0_variable` / `T0_noise` 参数
- [ ] `env/environment.py`：支持可变通信延迟
- [ ] `run.py test`：添加 `--t0` / `--t0-noise` 参数

**产出**：Fig.12 通信延迟敏感性曲线（延迟/成本/成功率 vs $T_0$）

---

### 论文新增图表清单

| 图表编号 | 内容 | 对应章节 | 优先级 |
|----------|------|----------|--------|
| Fig.4 (更新) | Sin + Twitter + Alibaba 负载曲线 | 5.1 | — |
| Fig.5 (更新) | PPO + SAC 训练收敛曲线（3 场景） | 5.2 | — |
| Fig.6 (更新) | 三指标柱状图（3 场景 × 6 agents） | 5.3 | — |
| Fig.7 (更新) | 延迟时隙变化（3 场景） | 5.3 | — |
| Fig.8 (更新) | 成本时隙变化（3 场景） | 5.3 | — |
| **Fig.9 (新增)** | 消融对比图 | 5.4 | P0 |
| **Fig.10 (更新/新增)** | V 参数敏感性（V=10/100/1000） | 5.5.1 | P1 |
| **Fig.11 (新增)** | 大规模可扩展性（性能 + 训练效率） | 5.5.2 | P1 |
| **Fig.12 (新增)** | 通信延迟敏感性 | 5.5.3 | P2 |
| **Table (新增)** | 训练开销表 | 5.2 | P0 |
| **Table (更新)** | 跨数据集性能汇总（Sin/Twitter/Alibaba） | 5.3 | P1 |
| **Table (新增)** | 消融结果汇总 | 5.4 | P0 |
| **Table (新增)** | 可扩展性汇总 | 5.5.2 | P1 |

---

## 3. Phase 0：最小复现验证

目标：用最小代价确认当前代码能复现论文数量级。

### 3.1 训练两个主场景 PPO

```bash
conda run -n as python run.py train \
    --config twitter_largescale --algo ppo --device cuda:0

conda run -n as python run.py train \
    --config sin_largescale --algo ppo --device cuda:1
```

预计耗时：large-scale 20000 epochs 每个约 4-6 小时。

### 3.2 测试原始 baseline

```bash
conda run -n as python run.py test \
    --config twitter_largescale \
    --model outputs/models/twitter_largescale/ppo/best \
    --agents NoScaling,Random,ProScaling,HPA,LGDRL \
    --output outputs/results/twitter_largescale/phase0

conda run -n as python run.py test \
    --config sin_largescale \
    --model outputs/models/sin_largescale/ppo/best \
    --agents NoScaling,Random,ProScaling,HPA,LGDRL \
    --output outputs/results/sin_largescale/phase0
```

### 3.3 通过标准

| 检查项 | 预期 | 判定标准 |
|--------|------|----------|
| 收敛性 | reward 经过 warm-up 后稳定 | 曲线不持续发散 |
| 排名 | LGDRL > ProScaling > HPA > Random > NoScaling | 大体排序一致 |
| Twitter 改进 | 接近论文幅度 | 偏差 < 50% 可接受 |
| Sin 改进 | 接近论文幅度 | 偏差 < 50% 可接受 |
| 成本 | 围绕预算 `C=35` | 平均成本在 25-45 |

如果 Phase 0 不通过，最多投入 2 天排查：奖励缩放、M/M/c 公式、队列截断、seed patch、config mismatch。

---

## 4. Phase 1：复现原论文图表

目标：在新增实验前先复现原始论文图表，形成可对照基线。

| 图表 | 内容 | 数据来源 | 状态 |
|------|------|----------|------|
| Fig.5 | Sin/Twitter 负载曲线 | `data/loads-*.txt` | 无需训练 |
| Fig.6 | 训练收敛曲线 | TensorBoard logs | 需 PPO 训练日志 |
| Fig.7 | 三指标平均性能 | Phase 0 测试结果 | 需模型和 baseline |
| Fig.8 | 延迟时隙变化 | `t_all.npy` | Phase 0 输出 |
| Fig.9 | 成本时隙变化 | `cost.npy` | Phase 0 输出 |
| Fig.10 | V 参数影响 | V=10/100/1000 模型 | 需额外训练 |

### 4.1 V 参数实验

V=100 为默认配置，额外训练 V=10、1000：

```bash
for V in 10 1000; do
    conda run -n as python run.py train \
        --config twitter_largescale --algo ppo --v "$V" --device cuda:0
done
```

注意：训练前必须确认模型保存路径能区分不同 V 值，避免 `outputs/models/twitter_largescale/ppo/best` 被覆盖。

测试：

```bash
for V in 10 100 1000; do
    conda run -n as python run.py test \
        --config twitter_largescale \
        --model outputs/models/twitter_largescale/ppo/best \
        --agents LGDRL \
        --output "outputs/results/twitter_largescale/V${V}"
done
```

---

## 5. Phase 2：P0 Major Revision 新增实验

这些实验优先完成，直接回应审稿人最核心的问题。

### 5.1 Lyapunov 消融实验（R1-3）

目标：量化 Lyapunov drift-plus-penalty reward 的贡献。

需要对比：

- [ ] Full AutoLFD/LGDRL
- [ ] w/o Lyapunov：奖励改为 `-(alpha * cost + beta * mean_delay)`
- [ ] w/o Historical Data：移除 state 历史到达率通道（ch 3-5），DRL 仅靠当前部署和资源状态决策
- [ ] 可选 w/o FFD：随机/中性初始部署，仅在实现稳定时加入

训练命令：

```bash
conda run -n as python run.py train \
    --config twitter_largescale --algo ppo \
    --ablation no_lyapunov --device cuda:0

conda run -n as python run.py train \
    --config sin_largescale --algo ppo \
    --ablation no_lyapunov --device cuda:1

conda run -n as python run.py train \
    --config twitter_largescale --algo ppo \
    --ablation no_prediction --device cuda:0
```

论文输出表：

| 变体 | 延迟 | 成本 | 成功率 | 结论 |
|------|------|------|--------|------|
| AutoLFD | TBD | TBD | TBD | 参考 |
| w/o Lyapunov | TBD | TBD | TBD | Lyapunov 贡献 |
| w/o Historical Data | TBD | TBD | TBD | 历史到达率信息贡献 |
| w/o FFD | TBD | TBD | TBD | 初始化贡献 |

### 5.2 预测机制回应（R2-3）

> **已合并至消融实验**。论文未提及具体预测模型（如 SMA），state 中历史到达率通道（ch 3-5）即为唯一预测信息来源。通过消融变体「w/o Historical Data」直接量化其贡献。预测精度（MAE/RMSE）等数值分析不设单独实验，在论文中以文字简要说明 SMA 窗口平均的合理性即可。

### 5.3 训练开销与部署成本（R2-5）

目标：量化 MDRL 的训练时间、收敛速度、推理延迟和 GPU 开销。

| 指标 | 数据来源 |
|------|----------|
| 总训练 wall time | logs/TensorBoard |
| 收敛所需 epochs | reward 曲线 |
| GPU memory peak | `torch.cuda.max_memory_allocated()` |
| 单步推理延迟 | action selection timer |
| 测试 rollout 时间 | runner timer |

实现任务：

- [ ] 在 `methods/PPO_dnn.py` 添加计时和显存统计。
- [ ] 若包含 SAC，对 `methods/SAC.py` 做同样统计。
- [ ] 保存到 `outputs/results/{config}/{experiment}/overhead.json`。

---

## 6. Phase 3：P1 扩展实验

这些实验能显著增强 Major Revision 说服力，但训练和实现成本更高。

### 6.1 新数据集验证（R2-4）

目标：在 Alibaba 或 Google trace 上验证泛化性。

当前优先选择 Alibaba：

- 数据：`data/loads-alibaba-v2022.txt`
- 配置：`alibaba_smallscale`, `alibaba_middlescale`, `alibaba_largescale`

任务：

- [ ] 确认 Alibaba 预处理方法，并在论文/附录说明数据来源。
- [ ] 运行 `python run.py list-configs`，确认配置可加载。
- [ ] 训练 `alibaba_largescale` PPO。
- [ ] 对所有 baselines 做测试。
- [ ] 将 Alibaba 结果加入主性能对比表。

命令：

```bash
conda run -n as python run.py train \
    --config alibaba_largescale --algo ppo --device cuda:0

conda run -n as python run.py test \
    --config alibaba_largescale \
    --model outputs/models/alibaba_largescale/ppo/best \
    --agents NoScaling,Random,ProScaling,HPA,LGDRL \
    --output outputs/results/alibaba_largescale/baseline
```

### 6.2 大规模可扩展性实验（R1-1）

目标：验证扁平动作空间在 50+ nodes 场景下的训练可行性和性能。

任务：

- [ ] 新增 `twitter_xlargescale.yaml` 或 `alibaba_xlargescale.yaml`。
- [ ] 先跑 30 nodes 作为显存/速度探针。
- [ ] 稳定后扩展到 50+ nodes。
- [ ] 报告动作空间规模、训练时间、显存和性能。

风险：50 nodes 可能超过 16GB 显存，或导致探索过稀疏。

兜底回应：若完整 50-node PPO 失败，报告可观察到的 scalability boundary，并在论文中讨论 hierarchical/parameterized action space。

### 6.3 DeepScaler/GNN baseline（R2-1）

目标：补充更强的图神经网络 autoscaling baseline。

参考：

- DeepScaler, ASE 2023：spatiotemporal GNN with adaptive graph learning。
- DeepScaling, SoCC 2022。
- GRAF, ToN 2024。

实现任务：

- [ ] 新增 `methods/DeepScalerGNN.py`。
- [ ] 使用 microservice call graph 和 edge-node resource graph。
- [ ] 尽量保持与 AutoLFD 相同 MDP/state/action surface。
- [ ] 在 `main.py` 和 `run.py` 注册 agent。
- [ ] 与 LGDRL 在相同 config 下对比。

建议对比：

| 对比 | 隔离因素 |
|------|----------|
| DeepScaler-GNN vs LGDRL | GNN baseline vs Lyapunov-guided DRL |
| DeepScaler-GNN vs LGDRL-CNN | 编码器差异 |
| 可选 LGDRL-GNN | Lyapunov reward + GNN encoder |

---

## 7. Phase 4：P2 分析与论文回应

### 7.1 通信延迟模型（R2-2）

优先方案：

- 将固定 `T0` 改为可配置的 variable latency model。
- 候选模型：base latency + hop latency + transfer size / bandwidth + bounded jitter。
- 在 `configs/base.yaml` 增加参数。
- 做小规模 sensitivity，不强制重跑所有实验。

保守方案：

- 在论文中承认固定 `T0` 的局限。
- 增加不同 `T0` 或随机扰动下的小规模敏感性分析。

### 7.2 历史数据利用分析（R1-2）

目标：说明和量化 `R_his` 对策略的贡献。

可选方法：

- Feature ablation：mask state channels 4-5 后测试性能下降。
- Gradient/saliency：分析不同 state channel 重要性。
- Encoder activation 或 t-SNE 可视化。

最低可接受产出：

- 文字解释：历史请求通道与部署、资源、预测通道一起输入 DRL encoder。
- 一个表格：mask 历史通道后的延迟/成本/成功率变化。

### 7.3 解耦框架 vs 端到端（R2-6）

回应思路：

- 承认 MFFD + MDRL 解耦可能带来最优性 gap。
- 强调解耦降低 action/state 联合优化难度，提高训练稳定性。
- 用消融或 GNN baseline 说明实际性能 gap 可控。
- 若不实现端到端模型，将其作为 future work。

---

## 8. 时间线

```text
Day 0-1
  Phase 0：训练/测试 twitter_largescale 和 sin_largescale。

Day 1
  决策点：如果排序和指标量级合理，继续 Phase 1。

Day 2-3
  Phase 1：复现原论文图表；补跑 V 参数实验。

Day 4-5
  Phase 2.1：Lyapunov 和 Historical Data 消融。

Day 6
  Phase 2.2：训练开销统计。

Day 7+
  Phase 3：Alibaba 验证、大规模实验、GNN baseline。

最后一周
  Phase 4：论文修改、response letter、图表和表格整理。
```

---

## 9. 风险清单

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Phase 0 无法复现论文趋势 | 后续实验失去基准 | 限时 2 天排查 reward、M/M/c、seed、config |
| V 参数模型互相覆盖 | Fig.10 无效 | 训练前确认模型路径包含 V 或实验名 |
| Gymnasium seed patch 缺失 | 结果不可复现 | 每次 batch run 前确认 patch |
| 大规模实验 OOM | R1-1 难完整回应 | 先跑 30 nodes，必要时报告 scalability boundary |
| DeepScaler 实现耗时 | R2-1 难完整回应 | 先实现简化 GNN baseline，再说明 adaptation |
| 通信模型改动影响所有指标 | 重跑成本高 | 优先做 sensitivity，不轻易改主实验环境 |

---

## 10. 输出目录约定

```text
outputs/
  models/
    {config}/{algo}/
    {config}/{algo}_{ablation}/
  results/
    {config}/{experiment}/
      data/{agent}/{metric}.npy
      overhead.json
  figures/
    fig05_load_curve.png
    fig06_convergence.png
    fig07_metrics.png
    fig08_latency_timeslot.png
    fig09_cost_timeslot.png
    fig10_v_parameter.png
    ablation_comparison.png
    training_overhead.png
    alibaba_comparison.png
    scalability.png
```

---

## 11. 最终交付物

- [ ] 修订论文正文。
- [ ] Response letter：逐条对应所有 reviewer comments。
- [ ] 原论文图表重生成。
- [ ] P0 新增图表/表格：
  - 消融实验对比（含 w/o Historical Data 回应 R1-2/R2-3）。
  - 训练开销。
- [ ] P1 图表/表格，如果完成：
  - Alibaba/Google trace 对比。
  - 大规模可扩展性。
  - DeepScaler/GNN baseline。
- [ ] 所有论文结果对应的可复现实验命令日志。
