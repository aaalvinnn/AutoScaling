# AGENTS.md — AutoLFD (AutoScaling)

Research codebase for the paper *"AutoLFD: A Three-Stage Framework for Microservice Fine-grained Auto-scaling in Edge Server Cluster"* (IEEE TCC, Major Revision, due 2026-07-02).

## Environment

- **Conda env**: `tcc` (Python 3.10, PyTorch 2.5.1, Gymnasium 1.0.0)
- **Hardware**: 2× RTX 4080 16GB
- **No build system** — no `pyproject.toml`, `setup.py`, `pytest`, or lint config.

## Critical: How to switch configuration

The **global `CONFIG`** used by both training and testing is set on **exactly one line**:

```
env/datastruct.py:10   CONFIG = config_alibaba_largescale.EnvConfig()
```

To switch scenarios, edit that import line. `main.py` and all `methods/*` files read it via `environment.CONFIG`. There is no CLI config selector — always check `datastruct.py` first when something doesn't match.

## How to run

```bash
# Training (uses CONFIG from datastruct.py, trains PPO agent)
conda run -n tcc python methods/PPO_dnn.py

# Testing / evaluation (uses CONFIG from datastruct.py)
conda run -n tcc python main.py

# Unit tests (manual, no framework)
conda run -n tcc python _unit_test.py
```

**Note**: The README says `method/PPO_dnn.py` (singular) but the actual directory is `methods/` (plural).

## Gymnasium patch

The installed `gymnasium` library was modified: `reset_async` distributes a single `int` seed to all parallel envs (instead of generating independent incremented seeds). If you reinstall gymnasium, you lose reproducibility — re-apply the patch from `README.md`.

## Architecture

| Directory | Purpose |
|---|---|
| `env/environment.py` | Gymnasium `DataCenterEnvironment` — M/M/c queue model, Lyapunov optimization, communication delay, reward computation |
| `env/datastruct.py` | Data structures (`MSInstance`, `Node`, `Request`, `TimeSlot`) + **global CONFIG** |
| `env/loghelper.py` | Test logging, matplotlib visualization, npy data dumps |
| `env/configs/` | One `EnvConfig` class per scenario (sin/twitter/alibaba × small/middle/large) |
| `methods/PPO_dnn.py` | Main algorithm: AutoLFD/LGDRL — CNN+DNN dual-input encoder, PPO training loop |
| `methods/SAC.py` | RL agent baseline (SAC with continuous action, separate from PPO) |
| `methods/FFD.py` | MFFD initial deployment strategy (First-Fit Decreasing) |
| `methods/Predicter.py` | SMA-based arrival-rate predictor |
| `methods/DeepScaler.py` | GNN baseline **skeleton** — not trained |
| `methods/HPA.py`, `ProScaling.py`, `GDCScaling.py`, `NoScaling.py`, `RandomScaling.py` | Baseline autoscaling policies |
| `model/` | Trained checkpoints + TensorBoard logs |
| `trained_models/` | Historical/baseline model snapshots |
| `test_output/` | Test results: `{config_name}/data/*.npy` + PNG charts |
| `data/` | Load traces: `loads-{sin,twitter,alibaba-v2022}.txt` |
| `draw_pictures/` | Paper figure generation (`fig1/`–`fig5/`) |
| `docs/` | Paper text, reviewer response letter |

## State and action spaces

**Observation**: `(7, ms_nums, server_node_nums)` — channels: deploy_info, cpus, memories, predicted_lamda, history_lamda (split across 2 channels), time_step_one_hot.

**Action (PPO/tuple)**: `(server_node, microservice, change_amount)` where `change_amount ∈ [-3, 3]`.

**Action (SAC/continuous)**: 3-dim Box, decoded to the same discrete semantics.

## Key parameters per config

- `V` — Lyapunov trade-off coefficient (default 0.1, historically 100: `Lyapunov Optimization` reward). Set via `self.V` in each config class.
- `delta` — queue truncation threshold.
- `C` — cost budget.
- `max_instance_update_num` — maximum instances to add/remove per step (default 3).
- Training: 20,000 epochs, 16 parallel envs, 288 timesteps per episode (24h / 5min slots).

## Current revision TODO (see `TODO.md` for full plan)

| Priority | Task | Status |
|---|---|---|
| P0 | Ablation: w/o Lyapunov, w/o Historical Data, w/o FFD | Not started |
| P0 | Training overhead stats (wall time, GPU, inference) | Not started |
| P1 | Alibaba trace full training + testing | Config exists, training not started |
| P1 | Large-scale (30/50+ nodes) | No configs yet |
| P1 | DeepScaler GNN baseline training | Code skeleton exists, not trained |

## Model save/load conventions

- Train output: `model/{config_name}/{MMDD}/{HHMM}/{algo}/` (date-time auto-generated)
- Test loads models from hardcoded paths in `main.py` (`LGDRL_MODEL_PATH`, `RLAGENT_MODEL_PATH` dicts)
- Checkpoints: `model_dnn_best.pth`, `model_dnn_{epoch}.pth`
- To test different checkpoints, edit the dict values in `main.py`

## Ablation flags

Ablation variants are controlled by modifying the environment's reward function or state construction: `w/o Lyapunov` replaces drift-plus-penalty with a weighted sum reward; `w/o Historical Data` removes history channels (3–5) from the observation; `w/o FFD` uses random initial deployment instead of MFFD. These are implemented by branching inside the environment/method code, not via CLI flags.

## Gotchas

- **Seed reproducibility**: `seed_all()` must be called before training. The config seed (1037) is the canonical seed.
- **State mutation**: `get_state()` returns a deep copy; the observation is reconstructed each step from `self.state` dict.
- **Queue stability**: `self.Qt` (virtual queue backlog) persists across reset in training loops — reset sets it to 0 only in `_reset_datastruct`.
- **Chinese comments**: Most comments are in Chinese.
- **Broken imports**: `_unit_test.py` imports `from env.config import EnvConfig` which doesn't exist (the directory is `env/configs/` not `env/config/`) — that import path only works because of `sys.path` hacks in `__init__.py` files, or it may fail.
