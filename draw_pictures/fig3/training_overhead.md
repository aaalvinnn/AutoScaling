# LGDRL Training Overhead & Inference Latency (twitter_largescale)

用于回应 Major Revision R2-5: Training Overhead。

## 实验环境

- GPU: 2× RTX 4080 16GB
- 框架: PyTorch 2.5.1 + Gymnasium 1.0.0
- 并行环境数: 16 (AsyncVectorEnv)
- 模型: `model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth`

## 训练开销

| 指标 | 数值 |
|------|------|
| 训练迭代 (iterations) | 10,000 |
| 总挂钟时间 | 17.6 小时 |
| 平均 SPS (steps/sec) | 1,453 |
| 模型参数量 | 657,948 (2.51 MB) |
| 每 iteration 批次大小 | 16 envs × 288 steps = 4,608 transitions |

数据来源: TensorBoard 日志 `model/twitter_largescale/0530/1829/PPO_dnn/events.out.tfevents.*`

## 推理延迟

每个 decision step 调用 `agent.get_action(state)` 的耗时（288 步测试），CPU 推理：

| 指标 | 数值 |
|------|------|
| Mean | 1.72 ms |
| Std | 1.72 ms |
| Median | 1.25 ms |
| P95 | 4.45 ms |
| P99 | 6.25 ms |

数据来源: `test_output/twitter_largescale/latency.npy`

## 提取方式

- 训练开销: `conda run -n tcc python scripts/extract_training_overhead.py`
- 推理延迟: `conda run -n tcc python main.py --config twitter_largescale`（自动输出并保存到 `test_output/`）
