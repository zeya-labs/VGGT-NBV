from __future__ import annotations

import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import CameraInput, compute_chamfer_record, compute_mesh_info, resolve_mesh_path


FRONTEND_DIR = BASE_DIR / "frontend"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = RESULTS_DIR / "index.json"

app = FastAPI()
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


class CameraSpec(BaseModel):
    position: List[float] = Field(..., min_length=3, max_length=3)
    target: Optional[List[float]] = Field(default=None, min_length=3, max_length=3)


class ComputeRequest(BaseModel):
    mesh_path: str
    cameras: List[CameraSpec]
    image_size: int = 256
    fov: float = 60.0


class MeshInfoResponse(BaseModel):
    mesh_path: str
    centroid: List[float]
    scale: float


class ComputeResponse(BaseModel):
    record_id: str
    mesh_path: str
    created_at: str
    loss_chamfer: float
    views: List[Dict[str, str]]
    cameras: List[CameraSpec]


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/mesh_info", response_model=MeshInfoResponse)
def mesh_info(path: str) -> MeshInfoResponse:
    mesh_path = resolve_mesh_path(path, REPO_ROOT)
    info = compute_mesh_info(mesh_path)
    return MeshInfoResponse(
        mesh_path=str(mesh_path),
        centroid=info.centroid,
        scale=info.scale,
    )


@app.get("/api/mesh_text")
def mesh_text(path: str) -> PlainTextResponse:
    mesh_path = resolve_mesh_path(path, REPO_ROOT)
    if not mesh_path.exists():
        raise HTTPException(status_code=404, detail=f"Mesh not found: {mesh_path}")
    try:
        text = mesh_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Mesh is not a text OBJ: {exc}")
    return PlainTextResponse(text)


@app.get("/api/history")
def history() -> List[Dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


@app.post("/api/history/clear")
def clear_history() -> Dict[str, int]:
    deleted_runs = 0
    if RESULTS_DIR.exists():
        for item in RESULTS_DIR.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
                deleted_runs += 1
            elif item.name == "index.json":
                item.unlink(missing_ok=True)
    return {"deleted_runs": deleted_runs}


@app.post("/api/calculate", response_model=ComputeResponse)
def calculate(payload: ComputeRequest) -> ComputeResponse:
    if not payload.cameras:
        raise HTTPException(status_code=400, detail="At least one camera is required")

    mesh_path = resolve_mesh_path(payload.mesh_path, REPO_ROOT)

    cameras = [
        CameraInput(position=c.position, target=c.target)
        for c in payload.cameras
    ]

    record_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    record_dir = RESULTS_DIR / record_id
    record_dir.mkdir(parents=True, exist_ok=True)

    try:
        record = compute_chamfer_record(
            mesh_path=mesh_path,
            cameras=cameras,
            output_dir=record_dir,
            image_size=payload.image_size,
            fov=payload.fov,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    record_payload = {
        "record_id": record_id,
        "mesh_path": str(mesh_path),
        "created_at": record.created_at,
        "loss_chamfer": record.loss_chamfer,
        "views": record.views,
        "cameras": [camera.dict() for camera in payload.cameras],
    }

    history_data = []
    if INDEX_PATH.exists():
        try:
            history_data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history_data = []
    history_data.append(record_payload)
    INDEX_PATH.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    return ComputeResponse(**record_payload)


__all__ = ["app"]
