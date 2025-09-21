# Repository Guidelines

## Project Structure & Module Organization
Core NBV logic lives in `nbv_framework/`, split into `models/` (policy networks + VGGT wrapper), `rendering/` (PyTorch3D renderer), `training/` (losses and trainer), `datasets/` (House3K and synthetic loaders), and `utils/` (camera, mesh, visualization helpers). The vendor `vggt/` tree hosts the pretrained VGGT model. Root-level `models/` stores geometry assets; treat it as read-only. Runtime artifacts land in `checkpoints/` and TensorBoard logs in `runs/`. Scripts like `train.py` and `test_depth.py` cover end-to-end workflows.

## Build, Test, and Development Commands
- `conda activate /mnt/sdb/chenmohan/env/vggt/` — enter the shared CUDA-ready environment with all dependencies preinstalled.
- `python train.py --mode train` — start policy training.
- `python train.py --mode eval` — run the evaluation loop on saved checkpoints.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, meaningful type hints, and concise module docstrings. Keep configuration dictionaries snake_case and pass them explicitly rather than relying on globals. Prefer descriptive class names (`BasicNBVPolicy`, `DifferentiableRenderer`) and prefix experimental scripts with `demo_` to distinguish them from library code.

## Testing Guidelines
Prefer GPU-enabled runs and verify deterministic seeds via `set_random_seed`. Extend integration coverage by adding scenario-specific scripts under the repository root or in a future `tests/` folder named `test_<module>.py`. Use `python train.py --mode all --no_auto_resume` for regression checks covering training and evaluation. When adding datasets, confirm loaders finish a dry run and capture key metrics in the PR notes.

## Commit & Pull Request Guidelines
Commit history follows Conventional Commit prefixes (`feat`, `fix`, `refactor`, etc.) with optional scopes, e.g. `feat(camera): add hemispheric sampler`. Keep summaries under 72 characters and describe motivation plus outcomes in the body. In PRs, link tracking issues, list required assets (dataset folders, checkpoints), and attach brief logs or TensorBoard screenshots for new training behaviour. Highlight any dependency upgrades or environment assumptions so reviewers can reproduce locally.

## Security & Configuration Tips
 Large geometry assets under `models/` and generated data under `synthetic_data/` should never be committed. Before sharing checkpoints, scrub paths and confirm no proprietary meshes leak through auxiliary files.
