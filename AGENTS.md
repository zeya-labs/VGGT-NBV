# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `nbv_framework/`: `models/` (VGGT/MapAnything wrappers and NBV policies), `rendering/` (PyTorch3D renderer), `training/` (Lightning trainers, Hydra config dataclasses), `datasets/` (synthetic data helpers), and `utils/` (camera, mesh, logging, data utilities).
- Hydra configs are in `configs/nbv/train.yaml`; override values via CLI (e.g., `python train.py learning_rate=5e-5 trainer.max_epochs=50`).
- Research experiments and smoke scripts sit in `experiments/`; reference implementations of upstream models are vendored under `vggt/`, `map-anything/`, and `Depth-Anything-V2/`.
- Generated artifacts: Lightning checkpoints and Wandb run files live under `runs/` (Hydra `output_dir`/`log_dir`), with scratch outputs in `outputs/` or `TEMP/`. Keep large files out of git.

## Build, Test, and Development Commands
- Activate the shared environment: `conda activate /mnt/sdb/chenmohan/env/mapanything/`.
- Training: `python -m train`.
- Resume or fine-tune: `python -m train resume_checkpoint=/path/to/ckpt.ckpt`.
- Visualize logs: use the Wandb UI (or run in offline mode and `wandb sync` later).

## Coding Style & Naming Conventions
- Python only: follow PEP 8 with 4-space indentation and descriptive snake_case for functions, modules, and Hydra keys. Use type hints (as in `NBVExperimentConfig`) and dataclasses for config structures.
- Keep logging consistent with `nbv_framework.utils.logging_utils.get_logger`; prefer structured log messages over print. The training stack is built on PyTorch Lightning + Hydra—follow their configuration idioms.
- Place small, focused helpers in `utils/`; avoid circular imports by keeping renderer/model logic inside their dedicated subpackages.

## Testing Guidelines
- There is no formal unit-test suite yet; run a fast sanity pass with Lightning: `python -m train`. This exercises data loading, model init, and a single training/val step.
- Keep deterministic runs by setting `seed` in the config and avoid changing `view_sampling_mode` unless measuring stochasticity.

## Commit & Pull Request Guidelines
- Follow the existing Conventional Commit pattern (`feat: ...`, `refactor(scope): ...`, `fix: ...`); include a scope when touching a specific area (`refactor(training): ...`).
- PRs should briefly state the experiment/config context, list any new Hydra parameters, and attach sample log metrics or screenshots (Wandb charts) when changing training behavior.
- Ensure scripts remain runnable: document new CLI flags in `AGENTS.md` or inline docstrings, and avoid committing generated checkpoints under `runs/`.
- Git Flow: branch from `develop` for features/bugs, open PRs back into `develop` (not `main`); keep branch names descriptive (e.g., `feature/nbv-loss-ablation`).

## Configuration & Data Hygiene
- Hydra changes that alter output paths (`output_dir`, `save_dir`, `log_dir`) must keep logs inside `runs/` to avoid polluting the repo root.
- Synthetic or downloaded assets belong under `outputs/` or `models/` when absolutely necessary, but training checkpoints stay in `runs/`. Use `.gitignore` for new cache paths when needed.
