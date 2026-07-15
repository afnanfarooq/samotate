"""Shared FastAPI dependencies."""
from __future__ import annotations

from fastapi import HTTPException, Request

from ..state import AppState, Project


def get_state(request: Request) -> AppState:
    return request.app.state.samotate


def get_project(request: Request, project_id: str) -> Project:
    project = get_state(request).store.get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"project '{project_id}' not found")
    return project
