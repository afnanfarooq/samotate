"""List project frames and serve raw image bytes."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from ...core.extract import list_frames
from ...models import Annotation
from .deps import frame_path, get_project

router = APIRouter(prefix="/api/projects", tags=["images"])


@router.get("/{project_id}/images")
def list_images(project=Depends(get_project)) -> list[dict]:
    cfg = project.config
    out = []
    for fp in list_frames(cfg):
        ann_path = cfg.annotations_dir / (fp.stem + ".json")
        reviewed = False
        n_boxes = 0
        if ann_path.exists():
            ann = Annotation.load(ann_path)
            reviewed = ann.reviewed
            n_boxes = len(ann.boxes)
        out.append({"name": fp.name, "reviewed": reviewed, "n_boxes": n_boxes})
    return out


@router.get("/{project_id}/images/{name}")
def get_image(name: str, project=Depends(get_project)) -> FileResponse:
    return FileResponse(frame_path(project.config, name))
