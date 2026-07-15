"""List project frames and serve raw image bytes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from ...core.extract import list_frames
from .deps import get_project

router = APIRouter(prefix="/api/projects", tags=["images"])


@router.get("/{project_id}/images")
def list_images(project=Depends(get_project)) -> list[dict]:
    cfg = project.config
    ann_dir = cfg.annotations_dir
    out = []
    for fp in list_frames(cfg):
        ann_path = ann_dir / (fp.stem + ".json")
        reviewed = False
        n_boxes = 0
        if ann_path.exists():
            from ...models import Annotation

            ann = Annotation.load(ann_path)
            reviewed = ann.reviewed
            n_boxes = len(ann.boxes)
        out.append({"name": fp.name, "reviewed": reviewed, "n_boxes": n_boxes})
    return out


@router.get("/{project_id}/images/{name}")
def get_image(name: str, project=Depends(get_project)) -> FileResponse:
    # prevent path traversal — only serve a real file directly in frames_dir
    fp = (project.config.frames_dir / name).resolve()
    frames_dir = project.config.frames_dir.resolve()
    if frames_dir not in fp.parents or not fp.is_file():
        raise HTTPException(404, "image not found")
    return FileResponse(fp)
