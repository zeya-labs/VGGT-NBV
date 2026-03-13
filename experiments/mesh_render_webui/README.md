# Mesh Render WebUI

Interactive mesh browser and PyTorch3D capture tool under `experiments/mesh_render_webui`.

## Features

- Browse meshes under `models/House3K_obj`
- Inspect OBJ and PLY meshes with orbit controls in the browser
- Capture a still image from the current viewer camera
- Render an MP4 video from the current viewer camera using PyTorch3D
- Save captures, videos, and metadata under `experiments/mesh_render_webui/results`

## Run

Activate the project environment first, then start the backend from repo root:

```bash
uvicorn experiments.mesh_render_webui.backend.app:app --reload --host 0.0.0.0 --port 8010
```

Open `http://127.0.0.1:8010`.

## Notes

- Default mesh: `models/House3K_obj/BATCH_1/Set_A/BAT1_SETA_HOUSE1.obj`
- Preview is rendered in-browser with Three.js using normalized geometry.
- Captured images and videos are rendered on the backend with PyTorch3D.

