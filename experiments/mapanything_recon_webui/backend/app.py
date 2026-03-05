from __future__ import annotations

import json
import logging
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
REPO_ROOT = PROJECT_DIR.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cache import PreparedRunCache
from pipeline import (
    discover_mesh_roots,
    list_mesh_files,
    path_to_repo_relative,
    prepare_inputs_for_run,
    reconstruct_and_export,
    resolve_mesh_root,
)
from schemas import (
    PrepareInputsRequest,
    PrepareInputsResponse,
    ReconstructRequest,
    ReconstructResponse,
    RunRecord,
)

logger = logging.getLogger("uvicorn.error")

FRONTEND_DIR = PROJECT_DIR / "frontend"
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = RESULTS_DIR / "index.json"

PREPARED_CACHE = PreparedRunCache(max_entries=8)

app = FastAPI(title="MapAnything Reconstruction WebUI")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")


def _model_dump(model) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _read_history() -> List[Dict[str, Any]]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _write_history(records: List[Dict[str, Any]]) -> None:
    INDEX_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def _upsert_history(run_id: str, payload: Dict[str, Any]) -> None:
    history = _read_history()
    updated = False
    for idx, record in enumerate(history):
        if record.get("run_id") == run_id:
            merged = dict(record)
            merged.update(payload)
            history[idx] = merged
            updated = True
            break
    if not updated:
        history.append(payload)
    _write_history(history)


@app.get("/")
def serve_index() -> FileResponse:
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.get("/api/mesh_roots")
def mesh_roots() -> Dict[str, Any]:
    roots = discover_mesh_roots(REPO_ROOT)
    return {
        "roots": [
            {
                "path": str(path),
                "label": path_to_repo_relative(path, REPO_ROOT),
            }
            for path in roots
        ]
    }


@app.get("/api/mesh_list")
def mesh_list(
    root: str | None = None,
    ext: str = "obj,ply",
    limit: int = Query(default=500, ge=1, le=20000),
) -> Dict[str, Any]:
    roots = discover_mesh_roots(REPO_ROOT)
    root_path = resolve_mesh_root(root, REPO_ROOT, roots)

    extensions = [item.strip() for item in ext.split(",") if item.strip()]
    files, truncated = list_mesh_files(
        root_path,
        extensions=extensions,
        limit=limit,
    )

    mesh_paths = [path_to_repo_relative(path, REPO_ROOT) for path in files]
    return {
        "root": str(root_path),
        "count": len(mesh_paths),
        "truncated": truncated,
        "meshes": mesh_paths,
    }


@app.post("/api/prepare_inputs", response_model=PrepareInputsResponse)
def prepare_inputs(payload: PrepareInputsRequest) -> PrepareInputsResponse:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = prepare_inputs_for_run(
            request=payload,
            run_id=run_id,
            created_at=created_at,
            output_dir=run_dir,
            results_dir=RESULTS_DIR,
            repo_root=REPO_ROOT,
        )
    except Exception as exc:
        logger.exception("prepare_inputs failed")
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    PREPARED_CACHE.put(result.prepared_run)

    record = RunRecord(
        run_id=run_id,
        created_at=created_at,
        mesh_path=result.prepared_run.mesh_path,
        model_name=result.prepared_run.model_name,
        num_views=payload.num_views,
        device=result.prepared_run.device_used,
        rgb_urls=result.rgb_urls,
        depth_urls=result.depth_urls,
        camera_poses=result.camera_poses,
        reconstruct=None,
    )
    _upsert_history(run_id, _model_dump(record))

    metadata = {
        "run_id": run_id,
        "created_at": created_at,
        "mesh_path": result.prepared_run.mesh_path,
        "model_name": result.prepared_run.model_name,
        "num_views": payload.num_views,
        "device": result.prepared_run.device_used,
        "timings": result.timings,
    }
    (run_dir / "prepare_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return PrepareInputsResponse(
        run_id=run_id,
        mesh_path=result.prepared_run.mesh_path,
        model_name=result.prepared_run.model_name,
        num_views=payload.num_views,
        rgb_urls=result.rgb_urls,
        depth_urls=result.depth_urls,
        camera_poses=result.camera_poses,
        created_at=created_at,
        device=result.prepared_run.device_used,
        timings=result.timings,
    )


@app.post("/api/reconstruct", response_model=ReconstructResponse)
def reconstruct(payload: ReconstructRequest) -> ReconstructResponse:
    prepared = PREPARED_CACHE.get(payload.run_id)
    if prepared is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Prepared run not found in memory cache. "
                "Please run /api/prepare_inputs again before reconstructing."
            ),
        )

    try:
        result = reconstruct_and_export(
            prepared_run=prepared,
            results_dir=RESULTS_DIR,
            conf_threshold=payload.conf_threshold,
            max_points=payload.max_points,
            use_depth_input=payload.use_depth_input,
            display_mode=payload.display_mode,
        )
    except Exception as exc:
        logger.exception("reconstruct failed for run_id=%s", payload.run_id)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    reconstruction_payload = {
        "ply_url": result.ply_url,
        "num_points": result.num_points,
        "num_points_before_sampling": result.num_points_before_sampling,
        "num_points_gt": result.num_points_gt,
        "num_points_recon": result.num_points_recon,
        "display_mode": result.display_mode,
        "timings": result.timings,
        "conf_threshold": payload.conf_threshold,
        "max_points": payload.max_points,
        "use_depth_input": payload.use_depth_input,
    }
    _upsert_history(payload.run_id, {"reconstruct": reconstruction_payload})

    (prepared.output_dir / "reconstruction" / "reconstruct_metadata.json").write_text(
        json.dumps(
            {
                "run_id": payload.run_id,
                **reconstruction_payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return ReconstructResponse(
        run_id=payload.run_id,
        ply_url=result.ply_url,
        num_points=result.num_points,
        num_points_before_sampling=result.num_points_before_sampling,
        num_points_gt=result.num_points_gt,
        num_points_recon=result.num_points_recon,
        display_mode=result.display_mode,
        used_depth_input=payload.use_depth_input,
        created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        timings=result.timings,
    )


@app.get("/api/history")
def history() -> List[Dict[str, Any]]:
    return _read_history()


@app.post("/api/history/clear")
def clear_history() -> Dict[str, int]:
    deleted_runs = 0
    for item in RESULTS_DIR.iterdir():
        if item.name in {"index.json", ".gitkeep"}:
            continue
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
            deleted_runs += 1
        elif item.is_file():
            item.unlink(missing_ok=True)

    INDEX_PATH.unlink(missing_ok=True)
    PREPARED_CACHE.clear()
    return {"deleted_runs": deleted_runs}
