"""Project CRUD and pipeline triggers (extract / detect / reconcile)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError

from ...config import ClassDef
from ...core import detect as detect_mod
from ...core import extract as extract_mod
from ...core import reconcile as reconcile_mod
from ...core.sam import SamNotAvailable
from ...errors import SamotateError
from .deps import get_project, get_state

router = APIRouter(prefix="/api/projects", tags=["projects"])


# Optional[...] rather than `X | None`: see the note in detect.py.
class CreateProjectRequest(BaseModel):
    name: str
    classes: list[ClassDef]
    video: Optional[str] = None
    images_dir: Optional[str] = None
    fps: float = Field(1.0, gt=0.0)
    limit: Optional[int] = Field(None, gt=0)


@router.get("")
def list_projects(request: Request) -> list[dict]:
    return [p.info() for p in get_state(request).store.list()]


@router.post("", status_code=201)
def create_project(request: Request, body: CreateProjectRequest) -> dict:
    try:
        project = get_state(request).store.create(
            name=body.name,
            classes=body.classes,
            video=body.video,
            images_dir=body.images_dir,
            fps=body.fps,
            limit=body.limit,
        )
    except ValidationError as exc:
        raise HTTPException(
            400, [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
        ) from exc
    return project.info()


@router.get("/{project_id}")
def get_project_info(project=Depends(get_project)) -> dict:
    return project.info()


@router.post("/{project_id}/extract")
def extract_frames(project=Depends(get_project)) -> dict:
    try:
        paths = extract_mod.gather_source_images(project.config)
    except SamotateError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"frames": len(paths)}


@router.post("/{project_id}/detect")
def detect(request: Request, project=Depends(get_project), force: bool = False) -> dict:
    cfg = project.config
    try:
        n = detect_mod.detect_all(cfg, backend=get_state(request).backend(), force=force)
    except SamNotAvailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except SamotateError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"detected": n}


@router.post("/{project_id}/reconcile")
def reconcile(project=Depends(get_project)) -> dict:
    try:
        n = reconcile_mod.reconcile_all(project.config)
    except SamotateError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"reconciled": n}
