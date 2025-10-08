# Repository Guidelines

## Project Structure & Module Organization
Core NBV logic lives in `nbv_framework/`, split into `models/` (policy networks + VGGT wrapper), `rendering/` (PyTorch3D renderer), `training/` (losses, trainer), `datasets/` (House3K and synthetic loaders), and `utils/` (camera, mesh, visualization helpers). Vendor code resides under `vggt/`. Geometry assets in root-level `models/` are read-only. Generated artifacts go to `checkpoints/` and TensorBoard logs land in `runs/`. Top-level scripts such as `train.py` and `test_depth.py` provide end-to-end workflows and demos.

## Build, Test, and Development Commands
Activate the shared environment with `conda activate /mnt/sdb/chenmohan/env/mapanything/`. Kick off training via `python train.py --mode train`; reuse the loop for evaluation with `python train.py --mode eval`. Run the regression combo using `python train.py --mode all --no_auto_resume`. Scripts expect CUDA; check `nvidia-smi` before long jobs.

## Coding Style & Naming Conventions
Follow PEP 8, four-space indentation, and explicit type hints. Keep configuration dictionaries snake_case and pass them explicitly instead of relying on globals. Use descriptive class names such as `BasicNBVPolicy` or `DifferentiableRenderer`. Prefix exploratory scripts with `demo_` to distinguish them from library code. Document non-obvious blocks with brief comments.

## Testing Guidelines
Prefer GPU-enabled runs and seed deterministically with `set_random_seed`. Place future integration tests under a `tests/` directory using `test_<module>.py` naming. For dataset additions, run a dry loader pass and capture key metrics alongside logs from `python train.py --mode all --no_auto_resume`. Include failure repro steps in PR notes whenever possible.

## Commit & Pull Request Guidelines
Use Conventional Commit prefixes, e.g., `feat(camera): add hemispheric sampler`, keeping summaries under 72 characters. PRs should reference related issues, list required assets or checkpoints, and attach concise training logs or TensorBoard screenshots for new behaviors. Highlight dependency or environment assumptions so reviewers can reproduce locally. Avoid committing large assets from `models/` or generated data from `synthetic_data/`.

## Security & Configuration Tips
Never commit geometry assets or generated datasets. Scrub checkpoint metadata for sensitive paths before sharing. Document any new environment variables or config knobs in the PR so other contributors can align their setups.
