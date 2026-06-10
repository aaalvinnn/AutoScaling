#!/usr/bin/env python3
"""
Extract LGDRL training overhead and inference latency from TensorBoard logs.

Usage:
    conda run -n tcc python scripts/extract_training_overhead.py
"""

import os
import sys
import glob
import torch
import numpy as np
from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── LGDRL model paths (from main.py LGDRL_MODEL_PATH) ──
LGDRL_MODEL_PATHS = {
    "sin_largescale":     "model/sin_largescale/0531/1359/PPO_dnn/model_dnn_best.pth",
    "twitter_largescale": "model/twitter_largescale/0530/1829/PPO_dnn/model_dnn_best.pth",
    "alibaba_largescale": "model/alibaba_largescale/0602/1440/PPO_dnn/model_dnn_best.pth",
}

# ── Ablation model paths (alibaba_largescale only) ──
ABLATION_PATHS = {
    "ablation_no_lyapunov_strict": "model/alibaba_largescale_no_lyapunov_strict/0605/1413/PPO_dnn/model_dnn_best.pth",
    "ablation_no_ffd":             "model/alibaba_largescale_no_ffd/0603/2043/PPO_dnn/model_dnn_best.pth",
    "ablation_no_history":         "model/alibaba_largescale_no_history/0603/2043/PPO_dnn/model_dnn_best.pth",
    "ablation_no_lyapunov":        "model/alibaba_largescale_no_lyapunov/0603/2043/PPO_dnn/model_dnn_best.pth",
}


def get_model_stats(path):
    """Return (param_count, size_mb) for a saved model checkpoint."""
    if not os.path.exists(path):
        return None, None
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        total = sum(v.numel() for v in sd.values())
        size_mb = sum(v.element_size() * v.numel() for v in sd.values()) / 1024 / 1024
        return total, size_mb
    except Exception:
        return None, None


def parse_tb_log(tb_dir):
    """Extract training statistics from a TensorBoard event directory."""
    if not os.path.isdir(tb_dir):
        return None

    # Find event files
    event_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))
    if not event_files:
        return None

    result = {}
    for ef in event_files:
        try:
            ea = EventAccumulator(os.path.dirname(ef))
            ea.Reload()
            scalars = ea.Tags().get("scalars", [])

            if "charts/reward" not in scalars:
                continue

            reward_events = ea.Scalars("charts/reward")
            if not reward_events:
                continue

            # Wall-clock time range
            first_wt = reward_events[0].wall_time
            last_wt = reward_events[-1].wall_time
            duration_s = last_wt - first_wt
            duration_h = duration_s / 3600

            # Iteration range
            total_iters = len(reward_events)
            first_iter = reward_events[0].step
            last_iter = reward_events[-1].step

            # SPS
            sps_vals = []
            if "charts/SPS" in scalars:
                sps_events = ea.Scalars("charts/SPS")
                sps_vals = [e.value for e in sps_events]

            # Use the event file with the most data points
            if total_iters > result.get("total_iters", 0):
                result = {
                    "total_iters": total_iters,
                    "first_iter": first_iter,
                    "last_iter": last_iter,
                    "duration_h": duration_h,
                    "duration_s": duration_s,
                    "start_time": datetime.fromtimestamp(first_wt).strftime("%Y-%m-%d %H:%M"),
                    "end_time": datetime.fromtimestamp(last_wt).strftime("%Y-%m-%d %H:%M"),
                    "sps_avg": sum(sps_vals) / len(sps_vals) if sps_vals else None,
                    "sps_min": min(sps_vals) if sps_vals else None,
                    "sps_max": max(sps_vals) if sps_vals else None,
                    "total_env_steps": last_iter,  # global_step at last logged iter
                }
        except Exception:
            continue

    return result if result else None


def main():
    print("=" * 90)
    print("LGDRL TRAINING OVERHEAD")
    print("=" * 90)

    header = f"{'Scenario':<20} {'Iters':>7} {'Time (h)':>9} {'SPS (avg)':>10} {'Params':>12} {'Size (MB)':>10}"
    print(header)
    print("-" * 90)

    for config_name, model_path in LGDRL_MODEL_PATHS.items():
        tb_dir = os.path.dirname(model_path)
        stats = parse_tb_log(tb_dir)
        param_count, size_mb = get_model_stats(model_path)

        if stats:
            iters = stats["total_iters"]
            time_h = f"{stats['duration_h']:.1f}"
            sps = f"{stats['sps_avg']:.0f}" if stats["sps_avg"] else "N/A"
        else:
            iters = "N/A"
            time_h = "N/A"
            sps = "N/A"

        params = f"{param_count:,}" if param_count else "N/A"
        size = f"{size_mb:.2f}" if size_mb else "N/A"
        print(f"{config_name:<20} {str(iters):>7} {time_h:>9} {sps:>10} {params:>12} {size:>10}")

    print()
    print("=" * 90)
    print("ABLATION (alibaba_largescale)")
    print("=" * 90)
    print(header)
    print("-" * 90)

    for ablation_name, model_path in ABLATION_PATHS.items():
        tb_dir = os.path.dirname(model_path)
        stats = parse_tb_log(tb_dir)
        param_count, size_mb = get_model_stats(model_path)

        if stats:
            iters = stats["total_iters"]
            time_h = f"{stats['duration_h']:.1f}"
            sps = f"{stats['sps_avg']:.0f}" if stats["sps_avg"] else "N/A"
        else:
            iters = "N/A"
            time_h = "N/A"
            sps = "N/A"

        params = f"{param_count:,}" if param_count else "N/A"
        size = f"{size_mb:.2f}" if size_mb else "N/A"
        print(f"{ablation_name:<20} {str(iters):>7} {time_h:>9} {sps:>10} {params:>12} {size:>10}")

    print()
    print("=" * 90)
    print("LGDRL INFERENCE LATENCY (from test_output/*/latency.npy)")
    print("=" * 90)
    latency_header = f"{'Scenario':<20} {'Mean (ms)':>10} {'Std (ms)':>10} {'Median (ms)':>12} {'P95 (ms)':>10} {'P99 (ms)':>10}"
    print(latency_header)
    print("-" * 90)

    for config_name in ["sin_largescale", "twitter_largescale", "alibaba_largescale"]:
        lat_path = os.path.join("test_output", config_name, "latency.npy")
        if os.path.exists(lat_path):
            data = np.load(lat_path, allow_pickle=True).item()
            if "LGDRL" in data:
                lats = np.array(data["LGDRL"])
                print(f"{config_name:<20} {np.mean(lats):>10.3f} {np.std(lats):>10.3f} "
                      f"{np.median(lats):>12.3f} {np.percentile(lats, 95):>10.3f} "
                      f"{np.percentile(lats, 99):>10.3f}")
            else:
                print(f"{config_name:<20} {'(no LGDRL key)':>10}")
        else:
            print(f"{config_name:<20} {'(not run yet)':>10}")


if __name__ == "__main__":
    main()
