# AGENTS.md — AutoLFD (AutoScaling)

Research codebase for the paper *"AutoLFD: A Three-Stage Framework for Microservice Fine-grained Auto-scaling in Edge Server Cluster"* (IEEE TCC, Major Revision, due 2026-07-02).

## Environment

- **Conda envs**: `tcc` (Python 3.10, PyTorch 2.5.1, Gymnasium 1.0.0) is the canonical env on the other machine; on **this** machine the working env is `as` (Python 3.10, PyTorch 2.11.0+cu130, Gymnasium 1.0.0, gymnasium patch verified). Commands below use `tcc` — substitute `as` locally.
- **Hardware**: 2× RTX 4080 16GB
- **No build system** — no `pyproject.toml`, `setup.py`, `pytest`, or lint config.

## Critical: Config switching mechanism

Config is resolved at **import time** via `env/datastruct.py` → `CONFIG = _resolve_config()`. The resolution order:

1. `--config <name>` CLI arg (parsed via raw `sys.argv` loop **before** any `from env import` — see `main.py:1-6`, `methods/PPO_dnn.py:24-28`)
2. `AUTOSCALING_CONFIG` env var
3. Default: `alibaba_largescale`

**Important**: Any file that does `from env import environment` (or imports `datastruct.py`) triggers config resolution. The `--config` argv parsing **must happen before those imports** or the config will be wrong.

Available configs: `sin_{small,middle,large}scale`, `twitter_{small,middle,large}scale`, `alibaba_largescale` — defined in `CONFIG_REGISTRY` dict in `datastruct.py`. There is also `config_twitter_largescale_requests15` (requests=15 variant) in the config directory but it's not in the registry — it must be imported and used directly.

## How to run

```bash
# Training (PPO — LGDRL main algorithm)
conda run -n tcc python methods/PPO_dnn.py --config twitter_largescale

# Training (SAC baseline)
conda run -n tcc python methods/SAC.py --config twitter_largescale

# Training (DeepScaler GNN baseline)
conda run -n tcc python methods/DeepScaler.py --config alibaba_largescale

# Testing / evaluation (currently 7-agent comparison)
conda run -n tcc python main.py --config twitter_largescale

# Unit tests (manual, no framework)
conda run -n tcc python _unit_test.py

# Training overhead extraction (TensorBoard + model stats)
conda run -n tcc python scripts/extract_training_overhead.py

# Export npy test results to Excel
conda run -n tcc python draw_pictures/export_excel.py
```

Models auto-save to `model/{config_name}/{MMDD}/{HHMM}/{algo}/`. Test loads from hardcoded paths in `main.py` dicts (`LGDRL_MODEL_PATH`, `RLAGENT_MODEL_PATH`, `DEEPSCALER_MODEL_PATH`).

## Ablation training

Use the standalone scripts (not CLI args — they hardcode alibaba config class):

```bash
# Full ablation training (10k epochs)
conda run -n tcc python train_ablation.py no_lyapunov_strict
conda run -n tcc python train_ablation.py no_history
conda run -n tcc python train_ablation.py no_ffd

# Resume from checkpoint
conda run -n tcc python resume_ablation.py no_lyapunov_strict model/.../model_dnn_5000.pth
```

These scripts set `config.config_name` directly (e.g. `"alibaba_largescale_no_lyapunov_strict"`) to change the model save directory. The `resume_ablation.py` only supports `no_lyapunov_strict` — extend it if you need other ablations.

## Gymnasium patch

The installed `gymnasium` was patched: `reset_async` distributes a single `int` seed identically to all parallel envs (instead of `[seed+i for i in ...]`). Reinstalling gymnasium breaks reproducibility — re-apply from `README.md`.

## Architecture

| Path | Purpose |
|---|---|
| `env/environment.py` | Gymnasium `DataCenterEnvironment` — M/M/c queue, Lyapunov optimization, reward computation |
| `env/datastruct.py` | Data structures + `CONFIG_REGISTRY` + `CONFIG` singleton (resolved at import) |
| `env/loghelper.py` | Test logging, matplotlib visualization, npy dumps |
| `env/configs/config_*.py` | One `EnvConfig` class per scenario — all hyperparams live here |
| `methods/PPO_dnn.py` | AutoLFD/LGDRL main algorithm (CNN+DNN encoder, PPO loop) |
| `methods/SAC.py` | SAC continuous-action baseline |
| `methods/DeepScaler.py` | GNN baseline (pure PyTorch, no external GNN lib) |
| `methods/FFD.py` | MFFD initial deployment |
| `methods/Predicter.py` | SMA-based arrival-rate predictor |
| `methods/{HPA,ProScaling,GDCScaling,NoScaling,RandomScaling}.py` | Baseline autoscaling policies |
| `draw_pictures/fig1-7/` | Paper figure generation scripts (each folder has `draw.py`) |
| `scripts/` | Utility scripts (`extract_training_overhead.py`) |
| `train_ablation.py` | Ablation training entry point (sets ablation flags + `config_name`) |
| `resume_ablation.py` | Resume ablation from checkpoint |
| `main.py` | Testing/evaluation entry point |
| `_unit_test.py` | Manual unit tests (no framework) |
| `model/` | Trained checkpoints + TensorBoard logs |
| `trained_models/` | Historical model snapshots (archived) |
| `test_output/` | Test results: `{config_name}/data/*.npy` + PNG charts + `latency.npy` |
| `data/` | Load traces: `loads-{sin,twitter,alibaba-v2022}.txt` |
| `docs/` | Paper text, review response letters |
| `release/` | Release artifacts |
| `test.ipynb` | Interactive test notebook |

### Abandoned / experimental methods

- `methods/DQN.py` — DQN implementation (no `--config` argv parsing, not usable with current config system)
- `methods/_SAC.py` — Older SAC variant
- `methods/_SAC_las.py` — SAC with lookahead state

These are not maintained. Ignore them unless you're explicitly reviving them.

### Sys.path hacks

`env/__init__.py` and `methods/__init__.py` both do `sys.path.append(dir_path)` so imports work from multiple working directories. `PPO_dnn.py`, `SAC.py`, and `DeepScaler.py` also do `sys.path.insert(0, parent_dir)` before importing env. This is fragile — don't rearrange import ordering.

## Ablation flags

Set as boolean attributes on the `EnvConfig` instance. Three flags exist in all large-scale configs:

- `ablation_no_lyapunov` / `ablation_no_lyapunov_strict` — replaces drift-plus-penalty with weighted-sum reward
- `ablation_no_history` — removes history channels from observation
- `ablation_no_ffd` — random initial deployment instead of MFFD

**Usage pattern** (from `draw_pictures/fig6/draw.py`):
```python
def make_config(**flags):
    c = config_alibaba_largescale.EnvConfig()
    for k, v in flags.items():
        setattr(c, k, v)
    return c
```

The environment reads these via `getattr(env_config, 'ablation_no_lyapunov_strict', False)` — so extra flags (like `_strict`) can be set without declaring them in the config class.

**Important**: After modifying an EnvConfig instance with ablation flags, you must assign `environment.CONFIG = cfg` before any agent or environment uses it. See `fig6/draw.py:54` for the pattern.

## Model save/load conventions

- Train output: `model/{config_name}/{MMDD}/{HHMM}/{algo}/` (auto-generated date-time dirs)
- Checkpoints: `model_dnn_best.pth`, `model_dnn_{epoch}.pth`
- Test loads from dicts in `main.py` (lines 123-151) — three dicts: `LGDRL_MODEL_PATH`, `RLAGENT_MODEL_PATH`, `DEEPSCALER_MODEL_PATH`. Edit dict values to switch checkpoints.
- Ablation models saved under `model/{config_name}_{ablation}/` (e.g., `model/alibaba_largescale_no_lyapunov_strict/`)
- DeepScaler saves as `model_{epoch}.pth` and `model_best.pth` (not `model_dnn_best.pth`)

## main.py testing setup

The active test configuration is at the bottom of `main.py` — currently a 7-agent comparison (NoScaling, Random, HPA, Proscale, RL Agent, LGDRL, DeepScaler) over 7 parallel envs. Previous test setups (V params, delta params, specific agents) are commented out above it. To change which agents are tested, edit the bottom section — you'll see the pattern of creating envs, loading agents, and constructing a `LogHelper`.

**DeepScaler loading quirk**: DeepScaler agents need `MS2MS_data_graph` from a temp environment. See `main.py:218-219`:
```python
temp_env = environment.DataCenterEnvironment(0, env_config, is_train=True)
temp_env.reset(seed=env_config.seed)
deepScalerAgent = DeepScaler.DeepScalerAgent(env_config, temp_env.MS2MS_data_graph)
```

`main.py` also measures inference latency (per-step `get_action` time) and saves it to `test_output/{config_name}/latency.npy`.

## Gotchas

- **Import order is load-bearing**: `--config` argv parsing happens before `from env import environment`. If you add new imports above the argv loop, config resolution will use the wrong scenario.
- **Mixed import paths**: `datastruct.py` uses both `from configs import ...` (relative) and `from env.configs import ...` (absolute). Both work due to `sys.path.append` in `__init__.py` files. Don't clean these up — it's fragile.
- **`_unit_test.py` broken import**: Line 2 does `from env.config import EnvConfig` — the directory is `env/configs/` not `env/config/`. May work via `sys.path` hacks or may fail depending on working directory.
- **Seed reproducibility**: `seed_all(1037)` must be called before training. The config seed is the canonical seed.
- **Queue stability**: `self.Qt` (virtual queue backlog) persists across `reset()` in training — `_reset_datastruct` zeros it but training loops manage it separately.
- **Chinese comments**: Most inline comments are in Chinese.
- **draw_pictures scripts are self-contained**: Each `fig*/draw.py` re-imports and re-runs the environment. They don't depend on `main.py` test output — they load model checkpoints directly and run fresh rollouts. They set `environment.CONFIG = cfg` before creating agents.
- **Config import from draw_pictures**: `fig6/draw.py` imports `from env.configs import config_alibaba_largescale` — this triggers config resolution. These scripts don't use `--config` argv; they manually overwrite `environment.CONFIG`.
- **DeepScaler config**: `DeepScaler.py` has its own `--config` argv parsing (lines 38-42), same pattern as PPO_dnn.py.
- **`config_alibaba_largescale.py` ablation flags**: This config only has 3 of the 4 flags — `ablation_no_lyapunov`, `ablation_no_history`, `ablation_no_ffd`. The `_strict` variant is set via `setattr` externally (it doesn't exist in the class).

## Current status

See `TODO.md` for the full task list. Key outstanding P0 items as of 2026-06-10:

- ❌ Training overhead stats for SAC and DeepScaler (LGDRL done — see `fig3/training_overhead.md`)
