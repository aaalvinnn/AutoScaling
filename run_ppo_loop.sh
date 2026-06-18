#!/bin/bash
cd /home/zsw/Papers/AutoScaling
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
while true; do
  echo "[$(date '+%H:%M:%S')] === PPO 启动 (auto-resume) ==="
  conda run -n as --no-capture-output python auto_resume_ppo_xlarge.py
  ec=$?
  echo "[$(date '+%H:%M:%S')] === PPO 退出 code=$ec, 5s 后续跑 ==="
  sleep 5
done
