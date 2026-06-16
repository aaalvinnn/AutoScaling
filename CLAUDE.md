# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research codebase for the paper *"AutoLFD: A Three-Stage Framework for Microservice Fine-grained Auto-scaling in Edge Server Cluster"* (IEEE TCC, under Major Revision, due 2026-07-02). The core contribution is an RL autoscaler (**LGDRL/AutoLFD** = PPO with a CNN+DNN dual-encoder over a Gymnasium M/M/c edge-server simulation) benchmarked against baselines (SAC, a GNN baseline, HPA, proportional, etc.).

> **`AGENTS.md` is the canonical, in-depth guide for this repo** — full architecture table, every gotcha, abandoned-method list, ablation-flag semantics. **Read it for anything beyond the quick reference below.** `README.md` has the bilingual project overview and setup; `TODO.md` tracks paper revision tasks.

## Environment

- **Conda envs — same codebase, two machines:**
  - `tcc` (Python 3.10, PyTorch 2.5.1, Gymnasium 1.0.0) — the canonical env on the other machine. **All commands below are written for `tcc`.**
  - `as` (Python 3.10, PyTorch 2.11.0+cu130, Gymnasium 1.0.0, gymnasium patch verified present) — the working env on **this** machine. Substitute `as` for `tcc` in any command here.
  - Run via `conda run -n <env> python ...` unless the env is already activated.
- **No build system, no test framework, no linter.** There is no `pytest`/`setup.py`/`pyproject.toml`. "Tests" = `_unit_test.py` (manual, you toggle which `test_*` runs at the bottom).
- **Hardware:** 2× RTX 4080 16GB. Training scripts hardcode `device` (often `cuda:1`); large-scale configs train on GPU.

### ⚠️ Gymnasium patch (reproducibility-critical)

`gymnasium/vector/async_vector_env.py` `reset_async` is hand-patched so a single `int` seed is broadcast *identically* to all parallel envs (instead of `seed + i`). **Reinstalling `gymnasium` silently breaks reproducibility** — re-apply the patch from `README.md` (the "⚠️ Gymnasium 补丁" section). Parallel-env seeds only agree because of this.

## Common commands

```bash
# Train the main algorithm (LGDRL = AutoLFD)
conda run -n tcc python methods/PPO_dnn.py --config twitter_largescale

# Train baselines
conda run -n tcc python methods/SAC.py --config twitter_largescale
conda run -n tcc python methods/DeepScaler.py --config alibaba_largescale   # GNN baseline, has its own --config parsing

# Evaluate — multi-agent comparison (7 agents). Edits model-path dicts at top of main.py first.
conda run -n tcc python main.py --config twitter_largescale

# Manual unit tests (no framework — toggle the active test_* at the bottom of the file)
conda run -n tcc python _unit_test.py

# Ablation training — arg1 picks the variant; optional arg2 picks the config (default alibaba_largescale)
conda run -n tcc python train_ablation.py no_lyapunov_strict                          # → alibaba_largescale_no_lyapunov_strict
conda run -n tcc python train_ablation.py no_lyapunov_strict twitter_largescale       # → twitter_largescale_no_lyapunov_strict
# variants: no_lyapunov, no_lyapunov_strict, no_history, no_ffd
conda run -n tcc python resume_ablation.py no_lyapunov_strict model/.../model_dnn_5000.pth  # resume only supports no_lyapunov_strict

# Paper figures — each draw_pictures/figN/draw.py is self-contained and must be run individually
conda run -n tcc python draw_pictures/fig1/draw.py
conda run -n tcc python draw_pictures/fig6/draw.py
# ... fig2–fig7 likewise

# Bundle all PDFs + READMEs into release/figures.zip
bash scripts/export_figures.sh

# Extract training overhead from TensorBoard logs + model stats
conda run -n tcc python scripts/extract_training_overhead.py
# Convert test npy results → Excel
conda run -n tcc python draw_pictures/export_excel.py
```

**Configs** (`--config <name>`, or `AUTOSCALING_CONFIG` env var): `sin_{small,middle,large}scale`, `twitter_{small,middle,large}scale`, `alibaba_largescale` (default). Defined in `CONFIG_REGISTRY` in [env/datastruct.py](env/datastruct.py).

## Architecture (the load-bearing parts)

**Config resolution is import-time and fragile.** [env/datastruct.py](env/datastruct.py) runs `CONFIG = _resolve_config()` at import. Resolution order: `--config` CLI arg → `AUTOSCALING_CONFIG` env var → default `alibaba_largescale`. **The `--config` argv parsing (a raw `sys.argv` loop) MUST run before any `from env import ...`** — see [main.py:1-6](main.py#L1-L6) and [methods/PPO_dnn.py:24-28](methods/PPO_dnn.py#L24-L28). If you add an import above that loop, the config silently resolves to the wrong scenario. This is the #1 footgun in the repo.

Data flow:
- **[env/environment.py](env/environment.py)** — `DataCenterEnvironment(gym.Env)`: M/M/c queue sim, request chains, Lyapunov drift-plus-penalty reward, virtual-queue backlog (`Qt`). Observation `(7, ms_nums, server_node_nums)`; PPO/DeepScaler action = `(node, microservice, change_amount)` with `change_amount ∈ [-max_update, max_update]`.
- **[env/configs/config_*.py](env/configs/)** — one `EnvConfig` dataclass-ish class per scenario holding *all* hyperparams (V, delta, C, network sizes, training lr/gamma/clip, num_envs, total_epoches). Edit the config class to tune anything.
- **[methods/PPO_dnn.py](methods/PPO_dnn.py)** — the main algorithm. CleanRL-style PPO. `train()` auto-saves to `model/{config_name}/{MMDD}/{HHMM}/PPO_dnn/` with TensorBoard logs.
- **[main.py](main.py)** — evaluation entry. The active experiment is configured at the very bottom (currently a 7-agent comparison); older setups (V-sweep, delta-sweep, subsets) are commented out above it. Also measures per-step inference latency → `test_output/{config_name}/latency.npy`.

**Model save/load conventions:**
- Training writes `model/{config_name}/{MMDD}/{HHMM}/{algo}/` with `model_dnn_best.pth` + `model_dnn_{epoch}.pth` (DeepScaler uses `model_best.pth` / `model_{epoch}.pth`).
- **Testing loads from hardcoded dicts in [main.py](main.py) (`LGDRL_MODEL_PATH`, `RLAGENT_MODEL_PATH`, `DEEPSCALER_MODEL_PATH`)** — one path per config_name. To test a new checkpoint, edit the dict value, not a CLI flag. `model/` and `trained_models/` are gitignored.
- Ablation models live under `model/{config_name}_{ablation}/` (e.g. `model/alibaba_largescale_no_lyapunov_strict/`).

**draw_pictures/figN/ scripts are self-contained.** Each `draw.py` re-imports the env, sets `environment.CONFIG = cfg` by hand (they don't use `--config`), loads model checkpoints from hardcoded paths inside the script, and runs fresh rollouts — they do **not** consume `main.py`'s `test_output/`. When adding/altering a figure, follow the `fig6/draw.py` `make_config(**flags)` pattern for ablation configs.

**Ablation flags** are plain attributes on an `EnvConfig` instance (`ablation_no_lyapunov`, `ablation_no_lyapunov_strict`, `ablation_no_history`, `ablation_no_ffd`); the environment reads them via `getattr(env_config, '...', False)`, so flags like `_strict` can be set with `setattr` without being declared. `train_ablation.py`/`resume_ablation.py` flip the flag and rewrite `config.config_name` (which reroutes the save dir) — see `AGENTS.md` § "Ablation flags".

## Gotchas (beyond AGENTS.md, the ones that bite most)

- **Don't reorder imports** in entry scripts — the argv loop before env-imports is load-bearing, and the `sys.path.append`/`insert` hacks in `env/__init__.py`, `methods/__init__.py`, and the three training scripts are what make mixed relative/absolute config imports (`from configs import ...` vs `from env.configs import ...`) both resolve. Fragile by design; don't "clean it up."
- **`seed_all(1037)`** must be called before training; the config seed is canonical.
- **DeepScaler needs a graph at load time** — `main.py` builds a throwaway `is_train=True` env just to extract `temp_env.MS2MS_data_graph` and passes it into `DeepScalerAgent(...)` (see [main.py:217-219](main.py#L217-L219)).
- **`_unit_test.py` has a typo'd import** (`from env.config import EnvConfig` — the dir is `env/configs/`); it works only via the sys.path hacks and may fail depending on cwd. Treat it as a scratch harness, not a reliable test suite.
- **Most inline comments are in Chinese.**
- **`TODO.md`** is the source of truth for current paper-revision status and pending experiments (training-overhead stats for SAC/DeepScaler are outstanding as of the last commit).
