"""Trigger YOLO export (labels + train/val split + data.yaml)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from ...core import export as export_mod
from .deps import get_project

router = APIRouter(prefix="/api/projects", tags=["export"])


class ExportRequest(BaseModel):
    every: int = 1
    offset: int = 0


@router.post("/{project_id}/export")
def export(body: ExportRequest, project=Depends(get_project)) -> dict:
    return export_mod.export_dataset(
        project.config, every=body.every, offset=body.offset
    )
