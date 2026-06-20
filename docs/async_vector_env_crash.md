# AsyncVectorEnv 训练崩溃排查记录

> 供排查 agent 使用。记录 `as` conda 环境下 PPO/DeepScaler/SAC 训练在 gymnasium `AsyncVectorEnv` 中反复崩溃的现象、错误、已试方案与待查方向。

## TL;DR

`as` 环境（numpy 2.2.6 + gymnasium 1.0.0）下，`AsyncVectorEnv` 的 **fork 子进程疑似发生内存/对象状态腐蚀**，训练在 60~1000 epoch 后随机崩溃。错误信息在正常 Python 语义下不应出现（`enumerate` 解包失败、整数索引失败、`self` 对象类型错乱），指向 fork 子进程状态损坏。

**2026-06-18 更新**：已确认 `AsyncVectorEnv(context='spawn')` 可以正常 pickle/start/reset/step；短程压力测试通过。当前三个训练入口已统一改成默认 `spawn` 后端，保留环境变量切换到 `sync/fork/forkserver` 做对照。该方案仍需完整长跑验证，但比 `SyncVectorEnv` 快，且避开 fork-after-CUDA/numpy 的主要风险面。

---

## 环境

- conda env `as`：Python 3.10, **numpy 2.2.6**, **gymnasium 1.0.0**, torch 2.11.0+cu130
- 硬件：2× RTX 4080 16GB，125GB RAM
- 仓库：`/home/zsw/Papers/AutoScaling`
- 场景：`twitter_xlargescale`（20 节点 × 20 微服务，`num_envs=8`）
- 注意：gymnasium 已打补丁——`async_vector_env.py` 的 `reset_async` 把单个 int seed 等同广播给所有 worker（见 `README.md` "⚠️ Gymnasium 补丁"）。重装 gymnasium 会丢补丁。

## 复现

```bash
conda run -n as python methods/PPO_dnn.py --config twitter_xlargescale
conda run -n as python methods/DeepScaler.py --config twitter_xlargescale
conda run -n as python methods/SAC.py --config twitter_xlargescale
```

三个算法都会崩，通常几十到一千 epoch 内。崩溃点都在 **env.step → `cal_total_access_delay` / `_update_arrival_rate`** 路径，且都在 **AsyncVectorEnv worker 子进程**里（gymnasium `_async_worker` → `env.step(data)`），主进程通过 `_raise_if_errors` 把 worker 异常重新抛出。

## 崩溃错误（多次运行采集，腐蚀位置每次不同）

所有错误都是 worker 子进程抛出，经 gymnasium 转发到主进程。**共同特征：正常 Python 下不可能发生 → 内存/对象腐蚀。**

### 错误 1 — `enumerate` 解包失败（PPO，无线程限制，epoch 740）
```
env/environment.py line 546, in _cal_request_success_rate
    for request, t in zip(self.RequestFlow_list, t_total_list):
ValueError: too many values to unpack (expected 0)
```
`zip` 产出 2-tuple，解包到 `request, t` 不应失败。"expected 0" 是 numpy 内部错误（`_wrapreduction`），疑似 `t_total_list`（numpy 数组）在腐蚀进程里 iteration 协议损坏。

### 错误 2 — 整数索引失败（DeepScaler，epoch 1005）
```
env/environment.py line 519, in _update_arrival_rate
    request.lamda = (request_lamda + lamda_random_matrix[i, self.timeslot.get_now()]) / 3
IndexError: only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`) and integer or boolean arrays are valid indices
```
`i`（enumerate 索引）和 `self.timeslot.get_now()`（int）本应是合法整数索引。报错说明其中一个变成了非法类型（numpy 异型标量？）。

### 错误 3 — `self` 类型错乱（DeepScaler，epoch 1005 另一 worker）
```
env/environment.py line 361, in _cal_execution_delay
    ms = self.MS_list[ms_id]
AttributeError: 'int' object has no attribute 'MS_list'
```
`self` 应是 DataCenterEnvironment，却变成 `int`。

### 错误 4 — `self` 类型错乱（PPO，线程限制 OMP=1，epoch 60）
```
env/environment.py line 520, in _update_arrival_rate
    request.lamda = (request_lamda + lamda_random_matrix[i, self.timeslot.get_now()]) / 3
AttributeError: 'MSInstance' object has no attribute 'timeslot'
```
`self` 应是 env，却变成 `MSInstance`。**注意：加了线程限制仍崩，且更快（60 vs 740）。**

### 错误 5 — numpy dtype 转换失败（SAC）
```
env/environment.py line 365, in _cal_execution_delay
    sum_img = np.sum(image_num_list)
TypeError: Cannot cast array data from dtype('float64') to dtype('bool') according to the rule 'safe'
```
`image_num_list`（部署向量）本应是整数，`np.sum` 不应触发 float→bool 转换。numpy 2.2 内部 `_wrapreduction` 异常。

## 根因假设

**numpy 2.2.6 + `AsyncVectorEnv` fork 腐蚀**。Linux 下 `AsyncVectorEnv` 默认用 `fork` 创建 worker。fork 时若父进程已加载 numpy/torch（含 BLAS 线程池、CUDA context），子进程继承到不一致的内存状态，导致对象属性/类型错乱、numpy 内部协议损坏。错误位置随机（腐蚀落到哪行就报哪行）。

支持证据：
- 同样代码在单 env（非向量化）跑 288 步随机动作 0 崩溃（烟雾测试通过）
- 错误在正常语义下不可能 → 不是逻辑 bug，是状态损坏
- `SyncVectorEnv`（无 fork）不崩（但慢 ~8×）
- 线程限制（`OMP_NUM_THREADS=1`）**未阻止**崩溃 → 不是单纯 BLAS 线程问题

## 已尝试的方案

| 方案 | 结果 |
|---|---|
| `environment.py` 给延迟函数加 try/except 返回 `T_max` | ❌ 治标不治本，腐蚀位置随机，包不住（如 `_update_arrival_rate` 没包） |
| `cal_total_access_delay` 强制 `np.array([float(x) ...])` | ❌ 只防 object 数组，防不住 `self`/索引腐蚀 |
| 线程限制 `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` | ❌ PPO 仍在 epoch 60 崩；且让 PPO/DeepScaler 慢 4-5×（单线程 numpy）。SAC 反而变快 |
| `SyncVectorEnv` | ✅ 不崩，但 env 步进串行慢 ~8×（PPO ~14s/it，~40h）。需配 `_reset_seed` 修复 |
| PPO auto-resume 循环（崩了 warm-start 续跑） | 🟡 能积累 best.pth，但单次仍崩，非根治 |
| `AsyncVectorEnv(context='spawn')` | ✅ **短测通过**。可 pickle/start/reset/step；20 轮 xlarge 随机动作 + CUDA 预初始化压力测试通过。仍需完整训练长跑验证 |

## 当前代码状态

**`environment.py` 已做的修改（保留中，对 Sync/Async 都无害）：**
1. `_reset_seed`：去掉 `assert self.seed == CONFIG.seed`，改为强制 `seed = CONFIG.seed`。原因：SyncVectorEnv 给各 env 分发 `seed+i` 触发原断言；强制锁定保持并行 env 初始场景一致（与打过补丁的 Async 同语义）。
2. `cal_total_access_delay`：`np.array([float(x) for x in t_exe_list], dtype=np.float64)` 强制 float64。
3. `_cal_execution_delay`：整体 try/except，异常返回 `request.T_max`；`np.sum(image_num_list)` 包了 `float()`。
4. `_cal_route_delay`：整体 try/except，异常返回 `request.T_max`；每跳延迟 `float(np.dot(...))`。

**三个训练脚本（`PPO_dnn.py` / `DeepScaler.py` / `SAC.py`）**：当前通过 `methods/vector_env_backend.py` 创建 vector env，默认：

```bash
AUTOSCALING_VECTOR_BACKEND=spawn
AUTOSCALING_VECTOR_SHARED_MEMORY=1
```

可临时切换：

```bash
# 最稳但慢：串行
AUTOSCALING_VECTOR_BACKEND=sync conda run -n as python methods/PPO_dnn.py --config twitter_xlargescale

# 复现旧问题：Linux 默认 fork
AUTOSCALING_VECTOR_BACKEND=fork conda run -n as python methods/PPO_dnn.py --config twitter_xlargescale

# spawn 但禁用 Gymnasium observation shared memory
AUTOSCALING_VECTOR_SHARED_MEMORY=0 conda run -n as python methods/PPO_dnn.py --config twitter_xlargescale
```

**临时脚本**：`auto_resume_ppo_xlarge.py` + `run_ppo_loop.sh`（PPO 自动续跑循环，Sync 方案下不需要）。

**诊断脚本**：`scripts/probe_vector_env_backend.py`

```bash
conda run -n as python scripts/probe_vector_env_backend.py --config twitter_xlargescale --backend spawn --iterations 20 --num-envs 8 --preinit-cuda
conda run -n as python scripts/probe_vector_env_backend.py --config twitter_xlargescale --backend fork --iterations 20 --num-envs 8 --preinit-cuda
```

2026-06-18 结果：`spawn/fork/sync` 两轮 smoke test 均通过；`spawn/fork` 在 CUDA 预初始化后 20 轮随机动作压力测试均通过。因此最小 env 随机步进尚未复现训练崩溃，问题更可能需要“长时间训练 + fork 后 CUDA/numpy/torch 组合状态”触发。

## 待排查方向（建议优先级）

1. **换 numpy 版本**：装个干净 env 试 `numpy==1.26.4` 或 `numpy<2.2`，看 Async 是否还崩。这是最快验证"是不是 numpy 2.2 的锅"。
   ```bash
   conda create -n as_test python=3.10 -y
   conda run -n as_test pip install numpy==1.26.4 gymnasium==1.0.0 torch ...
   # 复跑 PPO 看是否崩
   ```
2. **完整验证 `spawn` 训练**：跑 PPO/DeepScaler/SAC 到至少 1000 epoch，确认不再出现 worker 随机异常。
3. **最小复现继续加压**：`scripts/probe_vector_env_backend.py` 已覆盖纯 env+随机动作；若还要逼近训练，可增加 PPO/DeepScaler 网络推理与 optimizer step。
4. **gymnasium 补丁交互**：确认 `reset_async` 的 hand-patch 没有引入 worker 状态不一致（补丁只改 seed 广播，应无关，但值得 review）。
5. **torch CUDA context fork**：训练脚本 import torch 并初始化 CUDA 后才 fork env worker。试试在创建 `AsyncVectorEnv` **之前**不碰 CUDA，或用 `multiprocessing.set_start_method('spawn')` 全局切换。

## 关键文件

- `env/environment.py` — env 核心（step/延迟计算/seed），所有崩溃都在这
- `methods/vector_env_backend.py` — 训练脚本统一的 vector env 后端选择器
- `methods/PPO_dnn.py` / `methods/DeepScaler.py` / `methods/SAC.py` — 训练入口，调用统一后端选择器
- `scripts/probe_vector_env_backend.py` — 后端 smoke/stress 诊断脚本
- `env/configs/config_twitter_xlargescale.py` — 场景配置（20n×20ms）
- 日志样例：`logs/xlarge_{ppo,deepscaler,sac}.log`（含完整 worker 栈）
