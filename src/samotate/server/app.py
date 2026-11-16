"""FastAPI application factory for the headless REST API.

There is no bundled web UI; interactive API docs are served at ``/docs``.

No CORS middleware is installed on purpose. The API takes local filesystem
paths and serves image bytes back, so a permissive CORS policy would let any
website open in the user's browser read files through it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI

from .. import __version__
from .api import annotations, detect, export, images, projects
from .state import AppState


def create_app(
    config_path: Optional[str] = None,
    workspace: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``workspace`` is the directory that holds projects (one subdirectory per
    project, each with a ``config.yaml``). If omitted, it is derived from
    ``config_path``: ``serve -c ws/my-project/config.yaml`` uses ``ws/``.
    Otherwise it is the current directory.
    """
    if workspace is not None:
        ws = Path(workspace)
    elif config_path is not None:
        ws = Path(config_path).resolve().parent.parent
    else:
        ws = Path.cwd()

    app = FastAPI(title="samotate", version=__version__)
    app.state.samotate = AppState(ws)

    for module in (projects, images, annotations, detect, export):
        app.include_router(module.router)

    @app.get("/")
    def root() -> dict:
        return {"name": "samotate", "version": __version__, "docs": "/docs"}

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "workspace": str(ws)}

    return app
