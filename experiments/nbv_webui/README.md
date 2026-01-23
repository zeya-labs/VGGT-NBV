# NBV WebUI (experiments/nbv_webui)

A lightweight WebUI to explore OBJ models, capture camera viewpoints, and trigger the NBV chamfer pipeline.

## Features
- Server-side OBJ path input and 3D preview (OrbitControls).
- Add cameras from the current view (position + look-at target).
- Trigger differentiable rendering + GT point sampling + chamfer loss.
- Results are saved to disk and shown as a history list (RGB + point-map images).

## Tech Stack
- Frontend: Three.js (CDN, no build step)
- Backend: FastAPI + PyTorch3D + nbv_framework

## Setup
1. Install backend dependencies (use your existing project venv):
   ```bash
   cd experiments/nbv_webui/backend
   pip install -r requirements.txt
   ```

2. Start the server from the backend folder:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

3. Open in your browser:
   ```
   http://localhost:8000
   ```

## Usage
1. Enter a server-side OBJ path, e.g.:
   ```
   models/House3K_obj/BATCH_1/Set_A/BAT1_SETA_HOUSE1.obj
   ```
2. Click **Load** to preview the model.
3. Orbit to a good view, click **Add** to store the camera.
4. Click **Calculate** to run the backend pipeline.
5. Click **Clear** in the History panel to remove previous runs.
6. Results are listed in **History** and saved under:
   ```
   experiments/nbv_webui/results/
   ```

## Notes
- The backend normalizes meshes with the same `quantile` strategy as `nbv_framework/utils/mesh_utils.py`. The UI applies the same transform so camera positions match rendering.
- OBJ files must be text-based and accessible from the server (relative paths are resolved from the repo root).
- GPU is used automatically if available; CPU rendering is supported but slower.
