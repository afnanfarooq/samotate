"""Shared FastAPI dependencies and helpers."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException, Request

from ...config import ProjectConfig
from ..state import AppState, Project

# project ids are slugs (see state.slugify); rejecting anything else keeps
# ids like ".." from resolving outside the workspace
_PROJECT_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def get_state(request: Request) -> AppState:
    return request.app.state.samotate


def get_project(request: Request, project_id: str) -> Project:
    project = None
    if _PROJECT_ID.match(project_id):
        project = get_state(request).store.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"project '{project_id}' not found")
    return project


def frame_path(cfg: ProjectConfig, name: str) -> Path:
    """Resolve an image name to a file directly inside ``frames_dir``, or 404."""
    frames_dir = cfg.frames_dir.resolve()
    fp = (frames_dir / name).resolve()
    if fp.parent != frames_dir or not fp.is_file():
        raise HTTPException(404, "image not found")
    return fp
