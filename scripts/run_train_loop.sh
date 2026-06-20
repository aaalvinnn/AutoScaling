#!/bin/bash
# 通用训练重启循环：崩了自动从最近一次 train_state.pt 续跑（全状态）。
#
# 用法:
#   bash scripts/run_train_loop.sh ALGO CONFIG [GPU] [CONDA_ENV]
#     ALGO   = PPO_dnn | DeepScaler | SAC
#     CONFIG = twitter_xlargescale | twitter_largescale | ...
#     GPU    = 可选, 物理卡号 (SAC 硬编码 cuda:0 → 经 CUDA_VISIBLE_DEVICES 重映射)
#     CONDA_ENV = 可选, 默认 as
#
# 例:
#   bash scripts/run_train_loop.sh PPO_dnn    twitter_xlargescale 0
#   bash scripts/run_train_loop.sh DeepScaler twitter_xlargescale 0
#   bash scripts/run_train_loop.sh SAC        twitter_xlargescale 1
#
# 说明:
# - 不加线程限制 env (OMP_NUM_THREADS=1 等已证实挡不住崩溃且让 PPO 慢 4-5×)。
# - vector backend 默认 spawn (各训练脚本内部默认)；可用 AUTOSCALING_VECTOR_BACKEND=fork|sync 覆盖。
# - 日志重定向到 logs/loop_{ALGO}_{CONFIG}.log 方便 nohup 后台跑。

set -u

ALGO=${1:?usage: run_train_loop.sh ALGO CONFIG [GPU] [CONDA_ENV]}
CONFIG=${2:?}
GPU=${3:-}
ENV=${4:-as}

cd /home/zsw/Papers/AutoScaling || exit 1
mkdir -p logs

# GPU 重映射（SAC 硬编码 cuda:0，靠这个把物理 GPU1 映射成逻辑 cuda:0）
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

LOG="logs/loop_${ALGO}_${CONFIG}.log"
n=0
echo "[$(date '+%F %T')] === 启动重启循环: $ALGO / $CONFIG (env=$ENV gpu=$GPU) → $LOG ===" | tee -a "$LOG"

while true; do
    n=$((n + 1))
    echo "[$(date '+%F %T')] === #$n $ALGO/$CONFIG 启动 (auto-resume) ===" | tee -a "$LOG"
    conda run -n "$ENV" --no-capture-output python "methods/$ALGO.py" --config "$CONFIG" --auto-resume 2>&1 | tee -a "$LOG"
    ec=${PIPESTATUS[0]}
    # exit 0 = 训练正常完成（num_iterations 跑完）→ 停止循环；非 0 = 崩溃/被信号杀 → 续跑
    if [ "$ec" -eq 0 ]; then
        echo "[$(date '+%F %T')] === #$n $ALGO 正常完成 (exit 0)，停止重启循环 ===" | tee -a "$LOG"
        break
    fi
    echo "[$(date '+%F %T')] === #$n $ALGO 崩溃/异常 (code=$ec)，5s 后续跑 ===" | tee -a "$LOG"
    sleep 5
done
