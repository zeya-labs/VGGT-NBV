# MapAnything Reconstruction WebUI

Interactive experiment app to test `MapAnythingWrapper.reconstruct_and_evaluate` with inputs prepared by:
- `house3k_camera.py` sampling rules
- `house3k_sample_builder.py` sample format
- `batch_preparation_use_case.py` rendering/preparation flow

## Features
- Select mesh root and mesh path from a server-side dropdown list.
- Choose number of input images and camera sampling parameters.
- Preview rendered RGB images and optional depth visualizations.
- Run `reconstruct_and_evaluate` with optional depth input and display an interactive colored point cloud.
- Save reconstruction outputs (`.ply`) and metadata under `experiments/mapanything_recon_webui/results/`.

## Start
1. Activate your project environment.

2. Install WebUI backend dependencies:
```bash
cd experiments/mapanything_recon_webui/backend
pip install -r requirements.txt
```

3. Run server:
```bash
uvicorn app:app --host 0.0.0.0 --port 8010
```

4. Open in browser:
```text
http://localhost:8010
```

## API
- `GET /api/mesh_roots`
- `GET /api/mesh_list?root=...&ext=obj,ply&limit=...`
- `POST /api/prepare_inputs`
- `POST /api/reconstruct` (`use_depth_input=true|false`)
- `GET /api/history`
- `POST /api/history/clear`

## Output layout
- `results/<run_id>/inputs/*.png`
- `results/<run_id>/reconstruction/colored_point_cloud.ply`
- `results/<run_id>/prepare_metadata.json`
- `results/<run_id>/reconstruction/reconstruct_metadata.json`
- `results/index.json`

## Notes
- Mesh normalization defaults to `unit_sphere` in the frontend payload.
- `image_size` must be divisible by 14 for MapAnything's ViT patch embedding.
- The backend uses GPU automatically when CUDA is available.
- Prepared runs are cached in memory; if cache is evicted/restarted, re-run "Prepare Inputs" before reconstruction.
