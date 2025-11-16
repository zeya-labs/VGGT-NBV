# Repository Guidelines

## Project Structure & Module Organization
- `nbv_framework/` hosts the production code: `models/` (policy variants + VGGT wrapper), `rendering/` (PyTorch3D renderer), `training/` (trainer, loss, logging utilities), `datasets/` (House3K + synthetic loaders and samplers), and `utils/` (camera, evaluation, visualization helpers). Keep new framework code inside these modules.
- `experiments/` is for controlled probes such as `view1_camera_path_experiment.py` or Chamfer sweeps. Treat it as a staging ground for ideas—upstream the stable pieces into `nbv_framework/`.
- `docs/` contains MapAnything notes and logging guides. Update or add a doc whenever a workflow or config knob changes so the rest of the team can reproduce.
- `test/` currently stores repro cases and diagnostics. Future automated coverage belongs under a dedicated `tests/` directory using `test_<module>.py` files.
- Vendor and pretrained content lives under `vggt/`. Geometry assets in the root-level `models/` folder are read-only inputs and must never be modified.
- Generated artifacts flow to `checkpoints/`, streaming logs/TensorBoard events go to `runs/`, and ad-hoc dumps can use `outputs/`. Keep these folders out of commits.

## Build, Test, and Development Commands
1. Activate the shared environment: `conda activate /mnt/sdb/chenmohan/env/mapanything/`.
2. Standard training runs now rely on Hydra overrides: `python train.py mode=train create_data=true` (adds a small synthetic set when needed) or `python train.py mode=train resume_checkpoint=/path/to/ckpt` to continue from a checkpoint.
3. Evaluation/regression: `python train.py mode=eval resume_checkpoint=/path/to/ckpt` or `python train.py mode=all auto_resume=false` to execute train+eval in one sweep.
4. For multi-GPU or distributed jobs, rely on the built-in `init_distributed_mode` hook: `torchrun --nproc_per_node=<gpus> train.py mode=train auto_resume=false`.
5. Always verify CUDA availability with `nvidia-smi` before starting long runs and prefer running on the same driver/toolkit combo captured in `docs/日志使用指南.md`.
6. Place exploratory sweeps or ablations inside `experiments/` scripts so changes stay isolated from the main trainer.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation and explicit type hints for public functions/classes. Keep configuration dictionaries snake_case (`policy_hidden_dim`, `synthetic_data_root`) and pass them explicitly rather than relying on globals.
- Derive new policy or renderer variants from the existing abstractions (`BaseNBVPolicy`, `DifferentiableRenderer`) to keep trainer expectations consistent.
- Prefix short-lived helpers or CLI demos with `demo_` or host them under `experiments/`. Move anything that graduates to production quality into the appropriate `nbv_framework` submodule with accompanying docstrings.
- Add concise comments near non-obvious math blocks (e.g., camera normalization, distributed sync barriers) to reduce onboarding time for new contributors.

## Testing & Experiment Tracking
- Use `set_random_seed` (already wired inside `train.py`) and document the exact seed plus command line in PR descriptions. GPU-enabled integration runs are the default expectation.
- When extending datasets, run a dry loader pass and capture metrics via `python train.py mode=all auto_resume=false create_data=true`; attach the resulting log snippet from `runs/<experiment>` to the review.
- Reuse the built-in evaluation helpers (`evaluate_nbv_policy`, `compare_with_baselines`) and note any deviations or custom baselines in `docs/` so others can replicate.
- `experiments/` scripts should print reproducible stats (Chamfer, coverage, etc.) and log to `outputs/` with timestamps. Reference these logs when sharing findings.
- Add future automated tests under `tests/` (pytest-style) and keep the existing `test/` repro folders for manual debugging artifacts only.

## Branching, Commit & PR Workflow
- Always create a fresh branch before making changes; never commit directly to `develop`. Use `git checkout -b feature/<scope>-<short-desc>` or `bugfix/<issue>-<summary>` so reviewers can track work in progress.
- Follow Conventional Commits (`feat(camera): add hemispheric sampler`) and keep summaries ≤72 characters. Squash tiny WIP commits locally before opening a PR.
- PRs must reference related issues/tasks, call out required assets or checkpoints, and include concise logs or TensorBoard screenshots for behavior changes. List any new environment variables or config options you introduced.
- Avoid committing generated data (`models/synthetic_data/`, `outputs/`, `runs/`, `checkpoints/`). Large assets should be shared through the agreed storage channel and linked in the PR notes.

## Security & Configuration Tips
- Never check in geometry assets or derived datasets. Scrub checkpoint metadata for local paths or machine identifiers before sharing externally.
- Document new environment variables, secrets, or service endpoints inside `docs/` (and reference them in the PR) so collaborators can stay aligned without exposing sensitive data.
