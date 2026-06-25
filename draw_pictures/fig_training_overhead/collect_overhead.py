#!/usr/bin/env python3
"""
Collect the training overhead table for R2-5.

The primary table is fixed to twitter_largescale. Do not mix larger-scale runs
into that 10-node table; the Markdown output also appends the 20-node Twitter
comparison rows as a separate section.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


CONFIG_NAME = "twitter_largescale"
TWENTY_NODE_CONFIG_NAME = "twitter_xlargescale"
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
ARCHIVE_ROOT = Path("/home/zsw/Papers/AutoScaling_0619/AutoScaling")

LGDRL_RUN_DIR = ROOT / "model" / CONFIG_NAME / "0530" / "1829" / "PPO_dnn"
SAC_RUN_DIR = ARCHIVE_ROOT / "model" / CONFIG_NAME / "0531" / "1355" / "SAC"
DEEPSCALER_RUN_DIR = ARCHIVE_ROOT / "model" / CONFIG_NAME / "0607" / "1653" / "DeepScaler"
TEN_NODE_LATENCY_PATH = ARCHIVE_ROOT / "test_output" / CONFIG_NAME / "latency.npy"
TWENTY_NODE_RUN_DIR = ROOT / "model" / TWENTY_NODE_CONFIG_NAME / "0619" / "2040"
TWENTY_NODE_LATENCY_PATH = ROOT / "test_output" / TWENTY_NODE_CONFIG_NAME / "latency.npy"
# NOTE: use the cold-start run 0529/2157, NOT 0531/1359. 0531/1359 is a warm-start
# resume (its charts/y begins already-converged at y≈9.43 = the 0529/2157 end value),
# so its "convergence time" (1.41 h) is bogus. The real cold-start curve (41.4→9.4)
# converges at epoch 6361 ≈ 5.10 h. Run lives on the 0619 archive (not copied locally).
SIN_LGDRL_RUN_DIR = ARCHIVE_ROOT / "model" / "sin_largescale" / "0529" / "2157" / "PPO_dnn"
ALIBABA_LGDRL_RUN_DIR = ROOT / "model" / "alibaba_largescale" / "0602" / "1440" / "PPO_dnn"

RUNS = {
    "LGDRL": {
        "method": "AutoLFD (LGDRL)",
        "tb_dir": LGDRL_RUN_DIR,
        "model_path": LGDRL_RUN_DIR / "model_dnn_best.pth",
        "agent_type": "PPO",
        "latency_key": "LGDRL",
    },
    "SAC": {
        "method": "SAC",
        "tb_dir": SAC_RUN_DIR,
        "model_path": SAC_RUN_DIR / "model.pth",
        "agent_type": "SAC",
        "latency_key": "RL Agent",
    },
    "DeepScaler": {
        "method": "DeepScaler",
        "tb_dir": DEEPSCALER_RUN_DIR,
        "model_path": DEEPSCALER_RUN_DIR / "model_best.pth",
        "agent_type": "DeepScaler",
        "latency_key": "DeepScaler",
    },
}

TWENTY_NODE_RUNS = {
    "LGDRL": {
        "scenario": TWENTY_NODE_CONFIG_NAME,
        "method": "AutoLFD (LGDRL)",
        "tb_dir": TWENTY_NODE_RUN_DIR / "PPO_dnn",
        "model_path": TWENTY_NODE_RUN_DIR / "PPO_dnn" / "model_dnn_best.pth",
        "latency_key": "LGDRL",
    },
    "SAC": {
        "scenario": TWENTY_NODE_CONFIG_NAME,
        "method": "SAC",
        "tb_dir": TWENTY_NODE_RUN_DIR / "SAC",
        "model_path": TWENTY_NODE_RUN_DIR / "SAC" / "model.pth",
        "latency_key": "RL Agent",
    },
    "DeepScaler": {
        "scenario": TWENTY_NODE_CONFIG_NAME,
        "method": "DeepScaler",
        "tb_dir": TWENTY_NODE_RUN_DIR / "DeepScaler",
        "model_path": TWENTY_NODE_RUN_DIR / "DeepScaler" / "model_best.pth",
        "latency_key": "DeepScaler",
    },
}

PAPER_SUMMARY_RUNS = {
    "Sin": {
        "scenario": "sin_largescale",
        "tb_dir": SIN_LGDRL_RUN_DIR,
        "latency_path": ARCHIVE_ROOT / "test_output" / "sin_largescale" / "latency.npy",
        "latency_key": "LGDRL",
    },
    "Alibaba": {
        "scenario": "alibaba_largescale",
        "tb_dir": ALIBABA_LGDRL_RUN_DIR,
        "latency_path": ARCHIVE_ROOT / "test_output" / "alibaba_largescale" / "latency.npy",
        "latency_key": "LGDRL",
    },
    "Twitter": {
        "scenario": "twitter_largescale",
        "tb_dir": LGDRL_RUN_DIR,
        "latency_path": TEN_NODE_LATENCY_PATH,
        "latency_key": "LGDRL",
    },
    "Twitter(20)": {
        "scenario": "twitter_xlargescale",
        "tb_dir": TWENTY_NODE_RUN_DIR / "PPO_dnn",
        "latency_path": TWENTY_NODE_LATENCY_PATH,
        "latency_key": "LGDRL",
    },
}

OUTPUTS = {
    "json": DATA_DIR / "overhead_10node.json",
    "latency": DATA_DIR / "decision_latency_10node.npy",
    "md": OUT_DIR / "training_overhead_10node.md",
    "csv": OUT_DIR / "training_overhead_10node.csv",
    "tex": OUT_DIR / "training_overhead_10node.tex",
}
REQUIRED_SCALAR_TAGS = ("charts/y", "charts/SPS")


# Config resolution happens at import time in env/datastruct.py. Set this before
# importing env or methods modules.
os.environ["AUTOSCALING_CONFIG"] = CONFIG_NAME
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _read_varint(buf: bytes, i: int) -> tuple[int, int]:
    if not isinstance(buf, (bytes, bytearray, memoryview)) or not isinstance(i, int):
        raise EOFError("invalid varint buffer")
    result = 0
    shift = 0
    while i < len(buf):
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, i
        shift += 7
    raise EOFError("truncated varint")


def _iter_fields(buf: bytes):
    if not isinstance(buf, (bytes, bytearray, memoryview)):
        return
    i = 0
    n = len(buf)
    while i < n:
        try:
            tag, i = _read_varint(buf, i)
        except EOFError:
            break
        field_no = tag >> 3
        wire = tag & 7
        if wire == 0:
            try:
                value, i = _read_varint(buf, i)
            except EOFError:
                break
            yield field_no, wire, value
        elif wire == 1:
            if i + 8 > n:
                break
            yield field_no, wire, buf[i:i + 8]
            i += 8
        elif wire == 2:
            try:
                length, i = _read_varint(buf, i)
            except EOFError:
                break
            if i + length > n:
                break
            yield field_no, wire, buf[i:i + length]
            i += length
        elif wire == 5:
            if i + 4 > n:
                break
            yield field_no, wire, buf[i:i + 4]
            i += 4
        else:
            break


def _parse_summary_value(buf: bytes) -> tuple[str | None, float | None]:
    tag = None
    simple_value = None
    try:
        for field_no, wire, value in _iter_fields(buf):
            if field_no == 1 and wire == 2:
                tag = value.decode("utf-8", "replace")
            elif field_no == 2 and wire == 5:
                simple_value = struct.unpack("<f", value)[0]
    except Exception:
        return None, None
    return tag, simple_value


def read_tfevents_scalars(tb_dir: Path) -> dict[str, list[dict[str, float]]]:
    """Read scalar events with a small pure-Python TFEvent/protobuf parser."""
    event_files = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {tb_dir}")

    by_tag_step: dict[str, dict[int, tuple[float, float]]] = {}
    for event_file in event_files:
        buf = event_file.read_bytes()
        pos = 0
        n = len(buf)
        while pos + 12 <= n:
            try:
                length = int(struct.unpack("<Q", buf[pos:pos + 8])[0])
            except Exception:
                break
            record_start = pos + 12  # length + masked length CRC
            record_end = record_start + length
            next_pos = record_end + 4  # data + masked data CRC
            if record_end > n or next_pos > n:
                break
            event = buf[record_start:record_end]
            pos = next_pos

            wall_time = None
            step = None
            summary = None
            try:
                for field_no, wire, value in _iter_fields(event):
                    if field_no == 1 and wire == 1:
                        wall_time = struct.unpack("<d", value)[0]
                    elif field_no == 2 and wire == 0:
                        step = int(value)
                    elif field_no == 5 and wire == 2:
                        summary = value
            except Exception:
                continue

            if wall_time is None or step is None or summary is None:
                continue

            try:
                for field_no, wire, value in _iter_fields(summary):
                    if field_no != 1 or wire != 2:
                        continue
                    tag, scalar = _parse_summary_value(value)
                    if tag is None or scalar is None:
                        continue
                    by_tag_step.setdefault(tag, {})[step] = (wall_time, float(scalar))
            except Exception:
                continue

    result: dict[str, list[dict[str, float]]] = {}
    for tag, by_step in by_tag_step.items():
        result[tag] = [
            {"step": int(step), "wall_time": wall, "value": value}
            for step, (wall, value) in sorted(by_step.items())
        ]
    return result


def read_required_scalars(tb_dir: Path) -> dict[str, list[dict[str, float]]]:
    code = """
import json
import sys
from pathlib import Path

from draw_pictures.fig_training_overhead.collect_overhead import (
    REQUIRED_SCALAR_TAGS,
    read_tfevents_scalars,
)

scalars = read_tfevents_scalars(Path(sys.argv[1]))
print(json.dumps({tag: scalars.get(tag, []) for tag in REQUIRED_SCALAR_TAGS}))
"""
    last_proc = None
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-B", "-c", code, str(tb_dir)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        last_proc = proc
        time.sleep(0.5)

    stderr = (last_proc.stderr if last_proc else "").strip()
    raise RuntimeError(f"Failed to read TensorBoard scalars from {tb_dir}: {stderr}")


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(float)
    out = np.empty(len(values), dtype=float)
    cumsum = np.cumsum(np.insert(values.astype(float), 0, 0.0))
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = (cumsum[i + 1] - cumsum[start]) / (i - start + 1)
    return out


def convergence_stats(
    y_events: list[dict[str, float]],
    smooth_window: int,
    tail_fraction: float,
    tolerance_fraction: float,
) -> dict[str, Any]:
    if not y_events:
        raise ValueError("charts/y has no events")

    steps = np.array([e["step"] for e in y_events], dtype=int)
    wall_times = np.array([e["wall_time"] for e in y_events], dtype=float)
    y_values = np.array([e["value"] for e in y_events], dtype=float)
    smoothed = trailing_mean(y_values, smooth_window)

    tail_start = max(0, int(len(smoothed) * (1.0 - tail_fraction)))
    target = float(np.mean(smoothed[tail_start:]))
    initial = float(smoothed[0])

    if initial > target:
        threshold = target + tolerance_fraction * (initial - target)
        candidates = np.flatnonzero(smoothed <= threshold)
    else:
        tolerance = max(abs(target) * tolerance_fraction, 1e-9)
        threshold = target + tolerance
        candidates = np.flatnonzero(np.abs(smoothed - target) <= tolerance)

    idx = int(candidates[0]) if len(candidates) else len(smoothed) - 1
    total_training_time_h = float((wall_times[-1] - wall_times[0]) / 3600.0)
    convergence_time_h = float((wall_times[idx] - wall_times[0]) / 3600.0)

    return {
        "convergence_epoch": int(steps[idx]),
        "convergence_time_h": convergence_time_h,
        "total_training_time_h": total_training_time_h,
        "target_y": target,
        "threshold_y": float(threshold),
        "first_logged_epoch": int(steps[0]),
        "last_logged_epoch": int(steps[-1]),
        "num_logged_points": int(len(steps)),
        "smooth_window": int(smooth_window),
        "tail_fraction": float(tail_fraction),
        "tolerance_fraction": float(tolerance_fraction),
    }


def model_stats(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    code = """
import json
import sys
from pathlib import Path

import torch

model_path = Path(sys.argv[1])
state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
tensor_values = [v for v in state_dict.values() if torch.is_tensor(v)]
param_count = int(sum(v.numel() for v in tensor_values))
tensor_size_mb = float(sum(v.numel() * v.element_size() for v in tensor_values) / 1024 / 1024)
file_size_mb = float(model_path.stat().st_size / 1024 / 1024)
print(json.dumps({
    "param_count": param_count,
    "tensor_size_mb": tensor_size_mb,
    "file_size_mb": file_size_mb,
}))
"""
    last_proc = None
    for _ in range(3):
        proc = subprocess.run(
            [sys.executable, "-c", code, str(model_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return json.loads(proc.stdout)
        last_proc = proc
        time.sleep(0.5)

    stderr = (last_proc.stderr if last_proc else "").strip()
    raise RuntimeError(f"Failed to read model stats from {model_path}: {stderr}")


def scalar_mean(events: list[dict[str, float]]) -> float | None:
    if not events:
        return None
    return float(np.mean([e["value"] for e in events]))


def resolve_device(config_device: str, requested: str | None) -> str:
    import torch

    device = requested or config_device
    if not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        return "cpu"
    if ":" not in device:
        return device
    try:
        index = int(device.split(":", 1)[1])
    except ValueError:
        return "cuda:0"
    if index < torch.cuda.device_count():
        return device
    return "cuda:0"


def cuda_synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def measure_decision_latency(total_steps: int, device_override: str | None) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], str]:
    import torch

    from env import environment
    from methods import PPO_dnn, SAC, DeepScaler

    env_config = environment.CONFIG
    if env_config.config_name != CONFIG_NAME:
        raise RuntimeError(f"Expected {CONFIG_NAME}, got {env_config.config_name}")

    device = resolve_device(str(env_config.device), device_override)
    env_config.device = device
    PPO_dnn.CONFIG.device = device
    SAC.device = torch.device(device)

    PPO_dnn.seed_all(env_config.seed)

    temp_env = environment.DataCenterEnvironment(0, env_config, is_train=False)
    temp_env.reset(seed=env_config.seed)
    ms2ms_graph = temp_env.MS2MS_data_graph
    temp_env.close()

    agents = {
        "LGDRL": PPO_dnn.PPOAgent(env_config),
        "SAC": SAC.SACAgent(env_config),
        "DeepScaler": DeepScaler.DeepScalerAgent(env_config, ms2ms_graph),
    }
    agents["LGDRL"].load(str(RUNS["LGDRL"]["model_path"]))
    agents["LGDRL"].actorcrtic.eval()
    agents["SAC"].load(str(RUNS["SAC"]["tb_dir"]))
    agents["SAC"].actor.eval()
    agents["DeepScaler"].load(str(RUNS["DeepScaler"]["model_path"]))
    agents["DeepScaler"].actorcrtic.eval()

    latency_arrays: dict[str, np.ndarray] = {}
    latency_stats: dict[str, dict[str, Any]] = {}

    for key, agent in agents.items():
        agent_type = str(RUNS[key]["agent_type"])
        env = environment.DataCenterEnvironment(0, env_config, is_train=False, agent_type=agent_type)
        state, _ = env.reset(seed=env_config.seed)

        for _ in range(5):
            cuda_synchronize(device)
            agent.get_action(state)
            cuda_synchronize(device)

        state, _ = env.reset(seed=env_config.seed)
        latencies = []
        rollout_start = time.perf_counter()
        done = False
        for _ in range(total_steps):
            if done:
                break
            cuda_synchronize(device)
            t0 = time.perf_counter()
            action = agent.get_action(state)
            cuda_synchronize(device)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
            state, _, done, _, _ = env.step(action)
        rollout_time_s = time.perf_counter() - rollout_start
        env.close()

        arr = np.asarray(latencies, dtype=float)
        latency_arrays[RUNS[key]["method"]] = arr
        latency_stats[key] = {
            "decision_steps": int(len(arr)),
            "decision_mean_ms": float(np.mean(arr)),
            "decision_std_ms": float(np.std(arr)),
            "decision_median_ms": float(np.median(arr)),
            "decision_p95_ms": float(np.percentile(arr, 95)),
            "decision_p99_ms": float(np.percentile(arr, 99)),
            "rollout_time_s": float(rollout_time_s),
        }

    return latency_stats, latency_arrays, device


def collect_training_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    metadata: dict[str, Any] = {"sources": {}, "convergence_rule": {}}

    for key, run in RUNS.items():
        tb_dir = Path(run["tb_dir"])
        scalars = read_required_scalars(tb_dir)
        conv = convergence_stats(
            scalars.get("charts/y", []),
            smooth_window=args.smooth_window,
            tail_fraction=args.tail_fraction,
            tolerance_fraction=args.tolerance_fraction,
        )
        sps_avg = scalar_mean(scalars.get("charts/SPS", []))

        row = {
            "scenario": CONFIG_NAME,
            "method": run["method"],
            "training_epochs": conv["last_logged_epoch"],
            "convergence_epoch": conv["convergence_epoch"],
            "convergence_time_h": conv["convergence_time_h"],
            "total_training_time_h": conv["total_training_time_h"],
            "sps": sps_avg,
            "decision_mean_ms": None,
            "decision_p95_ms": None,
            "model_size_mb": None,
            "param_count": None,
            "checkpoint_file_size_mb": None,
            "first_logged_epoch": conv["first_logged_epoch"],
            "last_logged_epoch": conv["last_logged_epoch"],
        }
        rows.append(row)

        metadata["sources"][key] = {
            "tb_dir": relpath(tb_dir),
            "model_path": relpath(Path(run["model_path"])),
        }
        metadata["convergence_rule"][key] = conv

    return rows, metadata


def collect_static_training_rows(
    runs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    metadata: dict[str, Any] = {"sources": {}, "convergence_rule": {}}

    for key, run in runs.items():
        tb_dir = Path(run["tb_dir"])
        scalars = read_required_scalars(tb_dir)
        conv = convergence_stats(
            scalars.get("charts/y", []),
            smooth_window=args.smooth_window,
            tail_fraction=args.tail_fraction,
            tolerance_fraction=args.tolerance_fraction,
        )
        sps_avg = scalar_mean(scalars.get("charts/SPS", []))

        row = {
            "scenario": run["scenario"],
            "method": run["method"],
            "training_epochs": conv["last_logged_epoch"],
            "convergence_epoch": conv["convergence_epoch"],
            "convergence_time_h": conv["convergence_time_h"],
            "total_training_time_h": conv["total_training_time_h"],
            "sps": sps_avg,
            "decision_mean_ms": None,
            "decision_p95_ms": None,
            "model_size_mb": None,
            "param_count": None,
            "checkpoint_file_size_mb": None,
            "first_logged_epoch": conv["first_logged_epoch"],
            "last_logged_epoch": conv["last_logged_epoch"],
        }
        rows.append(row)

        metadata["sources"][key] = {
            "tb_dir": relpath(tb_dir),
            "model_path": relpath(Path(run["model_path"])),
        }
        metadata["convergence_rule"][key] = conv

    return rows, metadata


def attach_model_stats(rows: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> None:
    by_method = {run["method"]: run for run in runs.values()}
    for row in rows:
        stats = model_stats(Path(by_method[row["method"]]["model_path"]))
        row.update({
            "model_size_mb": stats["tensor_size_mb"],
            "param_count": stats["param_count"],
            "checkpoint_file_size_mb": stats["file_size_mb"],
        })


def attach_existing_latency(
    rows: list[dict[str, Any]],
    runs: dict[str, dict[str, Any]],
    latency_path: Path,
) -> dict[str, Any]:
    if not latency_path.exists():
        raise FileNotFoundError(latency_path)

    raw = np.load(latency_path, allow_pickle=True).item()
    latency_metadata: dict[str, Any] = {"path": relpath(latency_path), "keys": {}}
    by_method = {run["method"]: run for run in runs.values()}

    for row in rows:
        run = by_method[row["method"]]
        key = str(run["latency_key"])
        if key not in raw:
            raise KeyError(f"{latency_path} has no latency key '{key}' for {row['method']}")
        values = np.asarray(raw[key], dtype=float)
        row.update({
            "decision_steps": int(len(values)),
            "decision_mean_ms": float(np.mean(values)),
            "decision_std_ms": float(np.std(values)),
            "decision_median_ms": float(np.median(values)),
            "decision_p95_ms": float(np.percentile(values, 95)),
            "decision_p99_ms": float(np.percentile(values, 99)),
        })
        latency_metadata["keys"][row["method"]] = {
            "latency_key": key,
            "decision_steps": int(len(values)),
        }

    return latency_metadata


def load_latency_arrays_for_runs(
    runs: dict[str, dict[str, Any]],
    latency_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not latency_path.exists():
        raise FileNotFoundError(latency_path)

    raw = np.load(latency_path, allow_pickle=True).item()
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"path": relpath(latency_path), "keys": {}}
    for run in runs.values():
        key = str(run["latency_key"])
        if key not in raw:
            raise KeyError(f"{latency_path} has no latency key '{key}' for {run['method']}")
        values = np.asarray(raw[key], dtype=float)
        arrays[run["method"]] = values
        metadata["keys"][run["method"]] = {
            "latency_key": key,
            "decision_steps": int(len(values)),
        }
    return arrays, metadata


def attach_latency(rows: list[dict[str, Any]], latency_stats: dict[str, dict[str, Any]]) -> None:
    by_method = {RUNS[key]["method"]: stats for key, stats in latency_stats.items()}
    for row in rows:
        stats = by_method.get(row["method"])
        if not stats:
            continue
        row.update(stats)


def attach_latency_arrays(rows: list[dict[str, Any]], latency_arrays: dict[str, np.ndarray]) -> None:
    by_method = {row["method"]: row for row in rows}
    for method, values in latency_arrays.items():
        row = by_method.get(method)
        if row is None:
            continue
        arr = np.asarray(values, dtype=float)
        row.update({
            "decision_steps": int(len(arr)),
            "decision_mean_ms": float(np.mean(arr)),
            "decision_std_ms": float(np.std(arr)),
            "decision_median_ms": float(np.median(arr)),
            "decision_p95_ms": float(np.percentile(arr, 95)),
            "decision_p99_ms": float(np.percentile(arr, 99)),
        })


def collect_paper_summary_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    metadata: dict[str, Any] = {"sources": {}, "convergence_rule": {}, "latency": {}}

    for label, run in PAPER_SUMMARY_RUNS.items():
        tb_dir = Path(run["tb_dir"])
        scalars = read_required_scalars(tb_dir)
        conv = convergence_stats(
            scalars.get("charts/y", []),
            smooth_window=args.smooth_window,
            tail_fraction=args.tail_fraction,
            tolerance_fraction=args.tolerance_fraction,
        )

        latency_path = Path(run["latency_path"])
        raw = np.load(latency_path, allow_pickle=True).item()
        latency_key = str(run["latency_key"])
        if latency_key not in raw:
            raise KeyError(f"{latency_path} has no latency key '{latency_key}' for {label}")
        values = np.asarray(raw[latency_key], dtype=float)

        rows.append({
            "scenario_label": label,
            "scenario": run["scenario"],
            "training_time_h": conv["convergence_time_h"],
            "decision_mean_ms": float(np.mean(values)),
            "decision_p95_ms": float(np.percentile(values, 95)),
            "decision_steps": int(len(values)),
            "convergence_epoch": conv["convergence_epoch"],
            "training_epochs": conv["last_logged_epoch"],
        })
        metadata["sources"][label] = {
            "tb_dir": relpath(tb_dir),
            "latency_path": relpath(latency_path),
            "latency_key": latency_key,
        }
        metadata["convergence_rule"][label] = conv
        metadata["latency"][label] = {
            "decision_steps": int(len(values)),
            "decision_mean_ms": float(np.mean(values)),
            "decision_p95_ms": float(np.percentile(values, 95)),
        }

    return rows, metadata


def relpath(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def display_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        result.append({
            "Scenario": row["scenario"],
            "Method": row["method"],
            "Training Epochs": fmt(row["training_epochs"], 0),
            "Convergence Epoch": fmt(row["convergence_epoch"], 0),
            "Convergence Time (h)": fmt(row["convergence_time_h"], 2),
            "Total Training Time (h)": fmt(row["total_training_time_h"], 2),
            "SPS": fmt(row["sps"], 0),
            "Decision Mean (ms)": fmt(row["decision_mean_ms"], 3),
            "Decision P95 (ms)": fmt(row["decision_p95_ms"], 3),
            "Model Size": f"{fmt(row['model_size_mb'], 2)} MB",
        })
    return result


def display_paper_summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    result = []
    for row in rows:
        result.append({
            "Scenario": row["scenario_label"],
            "Training time (h)": fmt(row["training_time_h"], 2),
            "Decision Time (ms)": fmt(row["decision_mean_ms"], 2),
        })
    return result


def append_markdown_table(lines: list[str], rows: list[dict[str, Any]]) -> None:
    rendered = display_rows(rows)
    headers = list(rendered[0].keys())
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ])
    for row in rendered:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")


def append_rendered_markdown_table(lines: list[str], rendered: list[dict[str, str]]) -> None:
    headers = list(rendered[0].keys())
    lines.extend([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ])
    for row in rendered:
        lines.append("| " + " | ".join(row[h] for h in headers) + " |")


def write_markdown(
    rows: list[dict[str, Any]],
    path: Path,
    twenty_node_rows: list[dict[str, Any]] | None = None,
    paper_summary_rows: list[dict[str, Any]] | None = None,
) -> None:
    lines = [
        "# Training Overhead",
        "",
    ]
    if paper_summary_rows:
        lines.extend([
            "## Paper Summary",
            "",
            "This table reports AutoLFD convergence time and mean per-step decision time.",
            "",
        ])
        append_rendered_markdown_table(lines, display_paper_summary_rows(paper_summary_rows))
        lines.extend([
            "",
            "Training time is the convergence time detected from `charts/y`; decision time measures only `agent.get_action(state)` over 288 test slots.",
            "",
        ])

    lines.extend([
        "",
        "## 10-node Twitter",
        "",
        "Scenario: `twitter_largescale` (`node_nums=10`, `ms_nums=10`).",
        "",
    ])
    append_markdown_table(lines, rows)
    lines.extend([
        "",
        "Decision latency measures only `agent.get_action(state)` over 288 test slots; environment `step()` time is excluded.",
        "SAC and DeepScaler use archived 10-node runs under `/home/zsw/Papers/AutoScaling_0619/AutoScaling/model/twitter_largescale/`; decision latency comes from the matching archived `test_output/twitter_largescale/latency.npy`.",
        "",
    ])

    if twenty_node_rows:
        lines.extend([
            "## 20-node Twitter",
            "",
            "Scenario: `twitter_xlargescale` (`node_nums=20`, `ms_nums=20`).",
            "",
        ])
        append_markdown_table(lines, twenty_node_rows)
        lines.extend([
            "",
            "The 20-node rows use the `0619/2040` runs under `model/twitter_xlargescale/`, so AutoLFD keeps the same 3-hidden-layer architecture as the 10-node table; its model size is still larger because the state and action dimensions increase from 10x10 to 20x20.",
            "Decision latency comes from `test_output/twitter_xlargescale/latency.npy`; SAC is stored there under the `RL Agent` key.",
            "",
        ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    headers = list(display_rows(rows)[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(display_rows(rows))


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_tex(rows: list[dict[str, Any]], path: Path) -> None:
    rendered = display_rows(rows)
    headers = list(rendered[0].keys())
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Training and decision overhead on the 10-node Twitter scenario.}",
        r"\label{tab:training_overhead_10node}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        " & ".join(tex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rendered:
        lines.append(" & ".join(tex_escape(row[h]) for h in headers) + r" \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_paths(payload: dict[str, Any]) -> None:
    forbidden = "x" + "large"
    paths = []
    # Keep the original 10-node table guarded against accidental xlarge sources.
    # The 20-node Markdown-only reference rows are recorded separately on purpose.
    for source in payload.get("metadata", {}).get("sources", {}).values():
        paths.extend(source.values())
    paths.extend(relpath(path) for path in OUTPUTS.values())
    bad = [path for path in paths if forbidden in path]
    if bad:
        raise RuntimeError("Unexpected larger-scale path recorded: " + ", ".join(bad))


def validate_outputs(rows: list[dict[str, Any]], latency_arrays: dict[str, np.ndarray] | None, expected_steps: int) -> None:
    if len(rows) != len(RUNS):
        raise RuntimeError(f"Expected {len(RUNS)} rows, got {len(rows)}")
    for row in rows:
        required = [
            "scenario", "method", "training_epochs", "convergence_epoch", "convergence_time_h",
            "total_training_time_h", "sps", "model_size_mb",
        ]
        if latency_arrays is not None:
            required.extend(["decision_mean_ms", "decision_p95_ms"])
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise RuntimeError(f"{row.get('method')} missing fields: {missing}")
    if latency_arrays is not None:
        for method, values in latency_arrays.items():
            if len(values) != expected_steps:
                raise RuntimeError(f"{method} latency steps={len(values)}, expected {expected_steps}")


def validate_static_rows(rows: list[dict[str, Any]], require_latency: bool = False, expected_steps: int | None = None) -> None:
    required = [
        "scenario", "method", "training_epochs", "convergence_epoch", "convergence_time_h",
        "total_training_time_h", "sps", "model_size_mb",
    ]
    if require_latency:
        required.extend(["decision_mean_ms", "decision_p95_ms"])
    for row in rows:
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise RuntimeError(f"{row.get('method')} missing static fields: {missing}")
        if expected_steps is not None and row.get("decision_steps") != expected_steps:
            raise RuntimeError(
                f"{row.get('method')} latency steps={row.get('decision_steps')}, expected {expected_steps}"
            )


def validate_paper_summary_rows(rows: list[dict[str, Any]], expected_steps: int) -> None:
    if len(rows) != len(PAPER_SUMMARY_RUNS):
        raise RuntimeError(f"Expected {len(PAPER_SUMMARY_RUNS)} paper summary rows, got {len(rows)}")
    required = [
        "scenario_label", "scenario", "training_time_h", "decision_mean_ms",
        "decision_p95_ms", "decision_steps", "convergence_epoch", "training_epochs",
    ]
    for row in rows:
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise RuntimeError(f"{row.get('scenario_label')} missing paper summary fields: {missing}")
        if row.get("decision_steps") != expected_steps:
            raise RuntimeError(
                f"{row.get('scenario_label')} latency steps={row.get('decision_steps')}, expected {expected_steps}"
            )


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect training overhead table artifacts.")
    parser.add_argument("--total-steps", type=int, default=288, help="Number of test slots for decision timing.")
    parser.add_argument("--smooth-window", type=int, default=200, help="Trailing window for smoothed charts/y.")
    parser.add_argument("--tail-fraction", type=float, default=0.10, help="Final fraction used as stable target.")
    parser.add_argument("--tolerance-fraction", type=float, default=0.05, help="Convergence threshold fraction.")
    parser.add_argument("--device", help="Override inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--skip-decision", action="store_true", help="Only parse training logs; do not time get_action.")
    parser.add_argument("--remeasure-decision", action="store_true", help="Force rerunning get_action timing instead of reusing the saved latency file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows, metadata = collect_training_rows(args)
    twenty_node_rows, twenty_node_metadata = collect_static_training_rows(TWENTY_NODE_RUNS, args)
    paper_summary_rows, paper_summary_metadata = collect_paper_summary_rows(args)
    attach_model_stats(rows, RUNS)
    attach_model_stats(twenty_node_rows, TWENTY_NODE_RUNS)
    twenty_node_latency_metadata = attach_existing_latency(
        twenty_node_rows,
        TWENTY_NODE_RUNS,
        TWENTY_NODE_LATENCY_PATH,
    )
    latency_arrays = None
    decision_device = None
    if not args.skip_decision:
        if not args.remeasure_decision:
            latency_arrays, ten_node_latency_metadata = load_latency_arrays_for_runs(RUNS, TEN_NODE_LATENCY_PATH)
            attach_latency_arrays(rows, latency_arrays)
            np.save(OUTPUTS["latency"], latency_arrays, allow_pickle=True)
            decision_device = "archived_latency_file"
        else:
            latency_stats, latency_arrays, decision_device = measure_decision_latency(args.total_steps, args.device)
            attach_latency(rows, latency_stats)
            ten_node_latency_metadata = {
                "path": relpath(OUTPUTS["latency"]),
                "keys": {method: {"latency_key": method, "decision_steps": len(values)} for method, values in latency_arrays.items()},
            }
            np.save(OUTPUTS["latency"], latency_arrays, allow_pickle=True)
    else:
        ten_node_latency_metadata = None

    payload = {
        "config": CONFIG_NAME,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "decision_device": decision_device,
        "paper_summary_rows": paper_summary_rows,
        "paper_summary_metadata": paper_summary_metadata,
        "rows": rows,
        "extra_rows_20node": twenty_node_rows,
        "metadata": metadata,
        "latency_10node": ten_node_latency_metadata,
        "extra_metadata_20node": twenty_node_metadata,
        "extra_latency_20node": twenty_node_latency_metadata,
        "outputs": {key: relpath(path) for key, path in OUTPUTS.items()},
    }
    validate_paths(payload)
    validate_outputs(rows, latency_arrays, args.total_steps)
    validate_static_rows(twenty_node_rows, require_latency=True, expected_steps=args.total_steps)
    validate_paper_summary_rows(paper_summary_rows, expected_steps=args.total_steps)

    write_json(payload, OUTPUTS["json"])
    write_markdown(rows, OUTPUTS["md"], twenty_node_rows=twenty_node_rows, paper_summary_rows=paper_summary_rows)
    write_csv(rows, OUTPUTS["csv"])
    write_tex(rows, OUTPUTS["tex"])

    print(f"Wrote {relpath(OUTPUTS['json'])}")
    print(f"Wrote {relpath(OUTPUTS['md'])}")
    print(f"Wrote {relpath(OUTPUTS['csv'])}")
    print(f"Wrote {relpath(OUTPUTS['tex'])}")
    if latency_arrays is not None:
        print(f"Wrote {relpath(OUTPUTS['latency'])}")


if __name__ == "__main__":
    main()
