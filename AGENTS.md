# AGENTS.md — AutoLFD (AutoScaling)

Research codebase for the paper *"AutoLFD: A Three-Stage Framework for Microservice Fine-grained Auto-scaling in Edge Server Cluster"* (IEEE TCC, Major Revision, due 2026-07-02).

## Environment

- **Conda env**: `tcc` (Python 3.10, PyTorch 2.5.1, Gymnasium 1.0.0)
- **Hardware**: 2× RTX 4080 16GB
- **No build system** — no `pyproject.toml`, `setup.py`, `pytest`, or lint config.

## Critical: Config switching mechanism

Config is resolved at **import time** via `env/datastruct.py` → `CONFIG = _resolve_config()`. The resolution order:

1. `--config <name>` CLI arg (parsed via raw `sys.argv` loop **before** any `from env import` — see `main.py:1-6`, `methods/PPO_dnn.py:24-28`)
2. `AUTOSCALING_CONFIG` env var
3. Default: `alibaba_largescale`

**Important**: Any file that does `from env import environment` (or imports `datastruct.py`) triggers config resolution. The `--config` argv parsing **must happen before those imports** or the config will be wrong.

Available configs: `sin_{small,middle,large}scale`, `twitter_{small,middle,large}scale`, `alibaba_largescale` — defined in `CONFIG_REGISTRY` dict in `datastruct.py`.

## How to run

```bash
# Training (PPO)
conda run -n tcc python methods/PPO_dnn.py --config twitter_largescale

# Training (SAC baseline)
conda run -n tcc python methods/SAC.py --config twitter_largescale

# Training (DeepScaler GNN baseline)
conda run -n tcc python methods/DeepScaler.py --config alibaba_largescale

# Testing / evaluation
conda run -n tcc python main.py --config twitter_largescale

# Unit tests (manual, no framework)
conda run -n tcc python _unit_test.py
```

Models auto-save to `model/{config_name}/{MMDD}/{HHMM}/{algo}/`. Test loads from hardcoded paths in `main.py` dicts (`LGDRL_MODEL_PATH`, `RLAGENT_MODEL_PATH`, `DEEPSCALER_MODEL_PATH`).

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
| `draw_pictures/fig1-6/` | Paper figure generation scripts (each folder has `draw.py`) |
| `model/` | Trained checkpoints + TensorBoard logs |
| `test_output/` | Test results: `{config_name}/data/*.npy` + PNG charts |
| `data/` | Load traces: `loads-{sin,twitter,alibaba-v2022}.txt` |

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

## Model save/load conventions

- Train output: `model/{config_name}/{MMDD}/{HHMM}/{algo}/` (auto-generated date-time dirs)
- Checkpoints: `model_dnn_best.pth`, `model_dnn_{epoch}.pth`
- Test loads from dicts in `main.py` (lines 91-119) — edit dict values to switch checkpoints
- Ablation models saved under `model/{config_name}_{ablation}/` (e.g., `model/alibaba_largescale_no_lyapunov_strict/`)

## main.py testing setup

The active test configuration is at the bottom of `main.py` (lines 179-199). Previous test setups (V params, delta params, specific agents) are commented out above it. To change which agents are tested, edit the bottom section — you'll see the pattern of creating envs, loading agents, and constructing a `LogHelper`.

## Gotchas

- **Import order is load-bearing**: `--config` argv parsing happens before `from env import environment`. If you add new imports above the argv loop, config resolution will use the wrong scenario.
- **Mixed import paths**: `datastruct.py` uses both `from configs import ...` (relative) and `from env.configs import ...` (absolute). Both work due to `sys.path.append` in `__init__.py` files. Don't clean these up — it's fragile.
- **`_unit_test.py` broken import**: Line 2 does `from env.config import EnvConfig` — the directory is `env/configs/` not `env/config/`. May work via `sys.path` hacks or may fail depending on working directory.
- **Seed reproducibility**: `seed_all(1037)` must be called before training. The config seed is the canonical seed.
- **Queue stability**: `self.Qt` (virtual queue backlog) persists across `reset()` in training — `_reset_datastruct` zeros it but training loops manage it separately.
- **Chinese comments**: Most inline comments are in Chinese.
- **draw_pictures scripts are self-contained**: Each `fig*/draw.py` re-imports and re-runs the environment. They don't depend on `main.py` test output — they load model checkpoints directly and run fresh rollouts.

## Current revision status

See `TODO.md` for the full task list. Summary as of 2026-06-06:

- ✅ Alibaba trace: config, PPO/SAC/DeepScaler training, testing, all figures
- ✅ Ablation (alibaba only): w/o Lyapunov, w/o History, w/o FFD — models trained, fig6 generated
- ✅ Fig.1–6 all generated (three scenarios)
- ❌ Training overhead stats (R2-5) — need timer instrumentation
- ❌ Full baseline comparison across all agents/scenarios
