"""Trigger YOLO export (labels + train/val split + data.yaml)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ...core import export as export_mod
from ...errors import SamotateError
from .deps import get_project

router = APIRouter(prefix="/api/projects", tags=["export"])


class ExportRequest(BaseModel):
    every: int = Field(1, ge=1)
    offset: int = Field(0, ge=0)


@router.post("/{project_id}/export")
def export(body: ExportRequest, project=Depends(get_project)) -> dict:
    try:
        return export_mod.export_dataset(
            project.config, every=body.every, offset=body.offset
        )
    except SamotateError as exc:
        raise HTTPException(400, str(exc)) from exc
