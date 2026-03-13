from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
import sys
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline
from .schemas import (
    CaptureImageRequest,
    HistoryRecord,
    MeshInfoResponse,
    MeshListItem,
    RenderVideoRequest,
)


BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FRONTEND_DIR = BASE_DIR / "frontend"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = RESULTS_DIR / "index.json"

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Mesh Render WebUI")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


def _read_history() -> List[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_history(history: List[dict]) -> None:
    INDEX_PATH.write_text(json.dumps(history, indent=2), encoding="utf-8")


def _append_history(record: pipeline.CaptureRecord) -> dict:
    payload = {
        "record_id": record.record_id,
        "kind": record.kind,
        "mesh_path": record.mesh_path,
        "created_at": record.created_at,
        "preview_url": record.preview_url,
        "image_url": record.image_url,
        "video_url": record.video_url,
        "metadata_url": record.metadata_url,
    }
    history = _read_history()
    history.append(payload)
    _write_history(history)
    return payload


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/meshes", response_model=list[MeshListItem])
def get_meshes() -> list[MeshListItem]:
    meshes = pipeline.list_meshes(REPO_ROOT)
    return [MeshListItem(**mesh) for mesh in meshes]


@app.get("/api/mesh_info", response_model=MeshInfoResponse)
def get_mesh_info(path: str) -> MeshInfoResponse:
    try:
        mesh_path = pipeline.resolve_mesh_path(path, REPO_ROOT)
        info = pipeline.compute_mesh_info(mesh_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return MeshInfoResponse(
        mesh_path=str(mesh_path.relative_to(REPO_ROOT)),
        centroid=info.centroid,
        scale=info.scale,
    )


@app.get("/api/mesh_text")
def get_mesh_text(path: str) -> FileResponse:
    try:
        mesh_path = pipeline.resolve_mesh_path(path, REPO_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if mesh_path.suffix.lower() != ".obj":
        raise HTTPException(status_code=400, detail="mesh_text only supports OBJ files.")
    return FileResponse(mesh_path, media_type="text/plain; charset=utf-8")


@app.get("/api/mesh_asset")
def get_mesh_asset(path: str) -> FileResponse:
    try:
        mesh_path = pipeline.resolve_mesh_path(path, REPO_ROOT)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return FileResponse(mesh_path)


@app.get("/api/history", response_model=list[HistoryRecord])
def get_history() -> list[HistoryRecord]:
    return [HistoryRecord(**record) for record in _read_history()]


@app.post("/api/history/clear")
def clear_history() -> dict:
    deleted_runs = 0
    if RESULTS_DIR.exists():
        for item in RESULTS_DIR.iterdir():
            if item.name in {".gitkeep", ".gitignore", "index.json"}:
                continue
            if item.is_dir():
                shutil.rmtree(item)
                deleted_runs += 1
            elif item.is_file():
                item.unlink(missing_ok=True)
        INDEX_PATH.unlink(missing_ok=True)
    return {"deleted_runs": deleted_runs}


@app.post("/api/capture_image", response_model=HistoryRecord)
def capture_image(payload: CaptureImageRequest) -> HistoryRecord:
    try:
        mesh_path = pipeline.resolve_mesh_path(payload.mesh_path, REPO_ROOT)
        record = pipeline.capture_image_record(
            mesh_path=mesh_path,
            camera=pipeline.ViewerCameraSpec(
                position=list(payload.camera.position),
                target=list(payload.camera.target),
            ),
            output_root=RESULTS_DIR,
            image_size=int(payload.image_size),
            fov=float(payload.fov),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image capture failed for mesh %s", payload.mesh_path)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return HistoryRecord(**_append_history(record))


@app.post("/api/render_video", response_model=HistoryRecord)
def render_video(payload: RenderVideoRequest) -> HistoryRecord:
    try:
        mesh_path = pipeline.resolve_mesh_path(payload.mesh_path, REPO_ROOT)
        record = pipeline.render_video_record(
            mesh_path=mesh_path,
            camera=pipeline.ViewerCameraSpec(
                position=list(payload.camera.position),
                target=list(payload.camera.target),
            ),
            output_root=RESULTS_DIR,
            settings=pipeline.VideoRenderSettings(
                trajectory_mode=payload.trajectory_mode,
                duration_sec=float(payload.duration_sec),
                fps=int(payload.fps),
                image_size=int(payload.image_size),
                fov=float(payload.fov),
            ),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Video render failed for mesh %s", payload.mesh_path)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    return HistoryRecord(**_append_history(record))


__all__ = ["app"]

