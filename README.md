# AutoLFD

AutoLFD: A Three-Stage Framework for Microservice Fine-grained Auto-scaling in Edge Server Cluster

## 项目结构

```
AutoScaling/
├── main.py                  # 测试/评估入口
├── _unit_test.py            # 单元测试
├── env/
│   ├── environment.py       # Gymnasium 环境: M/M/c 排队模型, Lyapunov 优化, 奖励计算
│   ├── datastruct.py        # 数据结构 (MSInstance, Node, Request, TimeSlot) + 全局 CONFIG 注册表
│   ├── loghelper.py         # 测试日志、matplotlib 可视化、npy 数据导出
│   └── configs/             # 各场景参数配置
│       ├── config_sin_smallscale.py
│       ├── config_sin_middlescale.py
│       ├── config_sin_largescale.py
│       ├── config_twitter_smallscale.py
│       ├── config_twitter_middlescale.py
│       ├── config_twitter_largescale.py
│       └── config_alibaba_largescale.py
├── methods/
│   ├── PPO_dnn.py           # AutoLFD/LGDRL — CNN+DNN 双输入编码器, PPO 训练
│   ├── SAC.py               # SAC 连续动作空间基线
│   ├── DeepScaler.py        # GNN 基线 (纯 PyTorch, 无外部 GNN 库)
│   ├── FFD.py               # MFFD 初始部署策略 (First-Fit Decreasing)
│   ├── Predicter.py         # SMA 到达率预测器
│   ├── HPA.py               # HPA 基线
│   ├── ProScaling.py        # 比例缩放基线
│   ├── GDCScaling.py        # GDC 理想缩放基线
│   ├── NoScaling.py         # 无缩放基线
│   └── RandomScaling.py     # 随机缩放基线
├── data/                    # 负载轨迹数据
│   ├── loads-sin.txt
│   ├── loads-twitter.txt
│   └── loads-alibaba-v2022.txt
├── model/                   # 训练输出的模型 checkpoint + TensorBoard 日志
├── trained_models/          # 历史基线模型快照
├── test_output/             # 测试结果: {config_name}/data/*.npy + PNG 图表
├── draw_pictures/           # 论文图表生成 (fig1/–fig5/)
└── docs/                    # 论文文本、审稿意见回复
```

## 环境配置

```bash
conda create -n tcc python=3.10
conda activate tcc
pip install numpy gymnasium==1.0.0 torch==2.5.1 tensorboard tqdm matplotlib
```

### ⚠️ Gymnasium 补丁

本项目修改了 `gymnasium` 的 `AsyncVectorEnv.reset_async` 方法，使并行环境使用相同种子以保证可复现性。安装 gymnasium 后需要手动打补丁：

在 `gymnasium/vector/async_vector_env.py` 的 `reset_async` 方法中，将：
```python
elif isinstance(seed, int):
    seed = [seed + i for i in range(self.num_envs)]
```
替换为：
```python
elif isinstance(seed, int):
    seed = [seed for i in range(self.num_envs)]
```

> **重新安装 gymnasium 会覆盖此补丁，需重新应用。**

## 快速开始

所有入口脚本均支持 `--config <name>` 选择场景配置，也可通过环境变量 `AUTOSCALING_CONFIG` 设置。

### 可用配置

| 配置名 | 负载来源 | 节点数 | 微服务数 | 说明 |
|--------|---------|--------|---------|------|
| `sin_smallscale` | 正弦波 | 5 | 5 | 小规模 |
| `sin_middlescale` | 正弦波 | 8 | 8 | 中规模 |
| `sin_largescale` | 正弦波 | 10 | 10 | 大规模 |
| `twitter_smallscale` | Twitter Trace | 5 | 5 | 小规模 |
| `twitter_middlescale` | Twitter Trace | 8 | 8 | 中规模 |
| `twitter_largescale` | Twitter Trace | 10 | 10 | 大规模 |
| `alibaba_largescale` | Alibaba Trace v2022 | 10 | 10 | 大规模 |

不指定 `--config` 时默认使用 `alibaba_largescale`。

### 训练

```bash
# LGDRL (PPO + CNN+DNN)
conda run -n tcc python methods/PPO_dnn.py --config twitter_largescale

# SAC 基线
conda run -n tcc python methods/SAC.py --config twitter_largescale

# DeepScaler (GNN) 基线
conda run -n tcc python methods/DeepScaler.py --config twitter_largescale
```

训练输出的模型保存在 `model/{config_name}/{MMDD}/{HHMM}/{算法}/` 下，TensorBoard 日志在同一目录。

### 测试

```bash
conda run -n tcc python main.py --config twitter_largescale
```

测试前需在 `main.py` 中的 `LGDRL_MODEL_PATH` / `RLAGENT_MODEL_PATH` 字典里配置对应场景的模型路径。测试结果输出到 `test_output/{config_name}/`。

## 模型路径配置

`main.py` 中通过字典管理各场景的模型路径：

```python
LGDRL_MODEL_PATH = {
    "sin_smallscale": "trained_models/sin_smallscale/...",
    "twitter_largescale": "model/twitter_largescale/...",
    "alibaba_largescale": "model/alibaba_largescale/...",
    ...
}
```

切换场景测试时，`--config` 会自动匹配对应路径（通过 `env_config.config_name` 查字典）。

## 状态与动作空间

**观测 (Observation)**: `(7, ms_nums, server_node_nums)` — 通道依次为：部署信息、CPU、内存、预测到达率、历史到达率 (2 通道)、时隙 one-hot。

**动作 (PPO/离散)**: `(server_node, microservice, change_amount)`，其中 `change_amount ∈ [-3, 3]`。

**动作 (SAC/连续)**: 3 维 Box 空间，解码为相同的离散语义。

## 关键参数

各场景配置类中可调的主要参数：

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `V` | Lyapunov 权衡系数 | 100 |
| `delta` | 队列截断阈值 | 配置相关 |
| `C` | 时间平均开销预算 | 15–50 |
| `max_instance_update_num` | 每步最大实例增减数 | 2–3 |
| `num_envs` | 并行训练环境数 | 16–32 |
| `total_epoches` | 训练总轮数 | 8000–20000 |

训练超参数（`lr`, `gamma`, `clip_coef` 等）也在各配置类中定义，修改对应 `env/configs/config_*.py` 即可。

## 种子与可复现性

训练前需调用 `seed_all(seed)`，各配置默认使用 `seed = 1037`。配合上述 Gymnasium 补丁可保证并行环境种子一致。
