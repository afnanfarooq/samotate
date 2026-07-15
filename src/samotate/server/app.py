"""FastAPI application factory.

Mounts the API routers and serves the built Next.js export (if present) at the
root. The static bundle lives in ``server/static/`` and is produced by building
the ``frontend/`` app; when it's absent (e.g. a source checkout without a build)
the root returns a friendly message instead of 404.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .api import annotations, detect, export, images, projects
from .state import AppState

STATIC_DIR = Path(__file__).parent / "static"

_PLACEHOLDER = """\
<!doctype html><html><head><title>samotate</title></head>
<body style="font-family:sans-serif;max-width:40rem;margin:4rem auto">
<h1>samotate</h1>
<p>The backend is running, but the frontend has not been built.</p>
<p>From a source checkout, build it with:</p>
<pre>npm --prefix frontend install
npm --prefix frontend run build</pre>
<p>The API is available under <code>/api</code>.</p>
</body></html>
"""


def create_app(
    config_path: Optional[str] = None,
    workspace: Optional[str] = None,
) -> FastAPI:
    """Build the FastAPI app.

    ``workspace`` is the directory that holds projects. If omitted, it defaults
    to the parent of ``config_path`` (so ``serve -c project/config.yaml`` opens
    that project's workspace), or the current directory.
    """
    if workspace is not None:
        ws = Path(workspace)
    elif config_path is not None:
        ws = Path(config_path).resolve().parent.parent
    else:
        ws = Path.cwd()

    app = FastAPI(title="samotate", version="0.1.0")
    app.state.samotate = AppState(ws)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local-only tool; UI and API share an origin in prod
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for module in (projects, images, annotations, detect, export):
        app.include_router(module.router)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "workspace": str(ws)}

    if STATIC_DIR.exists() and (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:
        @app.get("/", response_class=HTMLResponse)
        def placeholder() -> str:
            return _PLACEHOLDER

    return app
