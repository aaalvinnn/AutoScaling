#!/usr/bin/env bash
set -u

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
CONFIG="${CONFIG:-twitter_xlargescale}"
CONDA_ENV="${CONDA_ENV:-as}"

export AUTOSCALING_VECTOR_BACKEND="${AUTOSCALING_VECTOR_BACKEND:-spawn}"
export AUTOSCALING_VECTOR_SHARED_MEMORY="${AUTOSCALING_VECTOR_SHARED_MEMORY:-1}"

mkdir -p logs

MASTER_LOG="logs/twitter_xlarge_methods_${RUN_ID}.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER_LOG"
}

run_method() {
    local name="$1"
    shift
    local log_file="logs/${RUN_ID}_${name}.log"

    log_msg "START ${name}: $*"
    "$@" >> "$log_file" 2>&1
    local status=$?
    log_msg "END ${name}: status=${status}, log=${log_file}"
    return "$status"
}

overall_status=0

log_msg "Sequential training run_id=${RUN_ID} config=${CONFIG} conda_env=${CONDA_ENV} backend=${AUTOSCALING_VECTOR_BACKEND} shared_memory=${AUTOSCALING_VECTOR_SHARED_MEMORY}"

run_method "ppo" conda run -n "$CONDA_ENV" python methods/PPO_dnn.py --config "$CONFIG" || overall_status=1
run_method "deepscaler" conda run -n "$CONDA_ENV" python methods/DeepScaler.py --config "$CONFIG" || overall_status=1
run_method "sac" conda run -n "$CONDA_ENV" python methods/SAC.py --config "$CONFIG" || overall_status=1

log_msg "DONE run_id=${RUN_ID} overall_status=${overall_status}"
exit "$overall_status"
