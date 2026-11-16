"""Server-side project store and shared state.

A *workspace* is a directory holding one subdirectory per project. Each project
subdirectory contains a ``config.yaml`` (a :class:`ProjectConfig`) and its
``dataset/`` working tree. The store is filesystem-backed — no database — which
keeps projects portable and inspectable.

The SAM backend is loaded lazily and shared across requests, so the (expensive)
model load happens once, on first assist use, not at startup.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Optional

from ..config import ClassDef, ProjectConfig
from ..core.extract import list_frames
from ..core.sam import SAMBackend


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "project"


class Project:
    """A project = its base directory + loaded config."""

    def __init__(self, project_id: str, base_dir: Path):
        self.id = project_id
        self.base_dir = base_dir
        self.config_path = base_dir / "config.yaml"

    @property
    def config(self) -> ProjectConfig:
        return ProjectConfig.from_yaml(self.config_path)

    def info(self) -> dict:
        cfg = self.config
        return {
            "id": self.id,
            "name": cfg.name,
            "source": "video" if cfg.video else "images",
            "classes": [c.model_dump() for c in cfg.classes],
            "n_frames": len(list_frames(cfg)),
        }


class ProjectStore:
    """Discovers, creates, and resolves projects within a workspace dir."""

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    def list(self) -> list[Project]:
        out = []
        for d in sorted(self.workspace.iterdir()):
            if d.is_dir() and (d / "config.yaml").exists():
                out.append(Project(d.name, d))
        return out

    def get(self, project_id: str) -> Optional[Project]:
        d = self.workspace / project_id
        if d.is_dir() and (d / "config.yaml").exists():
            return Project(project_id, d)
        return None

    def create(
        self,
        name: str,
        classes: list[ClassDef],
        *,
        video: Optional[str] = None,
        images_dir: Optional[str] = None,
        fps: float = 1.0,
        limit: Optional[int] = None,
    ) -> Project:
        project_id = slugify(name)
        base = self.workspace / project_id
        suffix = 2
        while base.exists():
            project_id = f"{slugify(name)}-{suffix}"
            base = self.workspace / project_id
            suffix += 1
        # validate before touching disk so a bad request leaves no empty dir
        cfg = ProjectConfig(
            name=name,
            video=Path(video) if video else None,
            images_dir=Path(images_dir) if images_dir else None,
            root=base / "dataset",
            fps=fps,
            limit=limit,
            classes=classes,
        )
        base.mkdir(parents=True)
        cfg.to_yaml(base / "config.yaml")
        return Project(project_id, base)


class AppState:
    """Shared application state attached to the FastAPI app."""

    def __init__(self, workspace: Path):
        self.store = ProjectStore(workspace)
        self._backend: Optional[SAMBackend] = None
        self._backend_lock = threading.Lock()

    def backend(self) -> SAMBackend:
        """Lazily load and cache the SAM backend (thread-safe)."""
        if self._backend is None:
            with self._backend_lock:
                if self._backend is None:
                    from ..core.sam import load_backend

                    # the API offers click prompts, so load the point head too
                    self._backend = load_backend(points=True)
        return self._backend
