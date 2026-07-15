"""Project CRUD and pipeline triggers (extract / detect)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...config import ClassDef
from ...core import detect as detect_mod
from ...core import extract as extract_mod
from .deps import get_project, get_state

router = APIRouter(prefix="/api/projects", tags=["projects"])


class CreateProjectRequest(BaseModel):
    name: str
    classes: list[ClassDef]
    video: str | None = None
    images_dir: str | None = None
    fps: float = 1.0
    limit: int | None = None


@router.get("")
def list_projects(request: Request) -> list[dict]:
    return [p.info() for p in get_state(request).store.list()]


@router.post("", status_code=201)
def create_project(request: Request, body: CreateProjectRequest) -> dict:
    if not body.video and not body.images_dir:
        raise HTTPException(400, "provide 'video' or 'images_dir'")
    project = get_state(request).store.create(
        name=body.name,
        classes=body.classes,
        video=body.video,
        images_dir=body.images_dir,
        fps=body.fps,
        limit=body.limit,
    )
    return project.info()


@router.get("/{project_id}")
def get_project_info(project=Depends(get_project)) -> dict:
    return project.info()


@router.post("/{project_id}/extract")
def extract_frames(project=Depends(get_project)) -> dict:
    paths = extract_mod.gather_source_images(project.config)
    return {"frames": len(paths)}


@router.post("/{project_id}/detect")
def detect(request: Request, project=Depends(get_project), force: bool = False) -> dict:
    backend = get_state(request).backend()
    n = detect_mod.detect_all(project.config, backend=backend, force=force)
    return {"detected": n}
