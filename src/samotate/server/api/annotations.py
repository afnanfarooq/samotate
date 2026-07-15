"""Get and save per-image annotations."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...core.extract import IMAGE_EXTS
from ...models import Annotation
from .deps import get_project

router = APIRouter(prefix="/api/projects", tags=["annotations"])


def _frame_path(project, name: str):
    fp = (project.config.frames_dir / name).resolve()
    frames_dir = project.config.frames_dir.resolve()
    if frames_dir not in fp.parents or not fp.is_file():
        raise HTTPException(404, "image not found")
    return fp


def _image_size(fp):
    from PIL import Image

    with Image.open(fp) as im:
        return im.width, im.height


@router.get("/{project_id}/images/{name}/annotation")
def get_annotation(name: str, project=Depends(get_project)) -> dict:
    fp = _frame_path(project, name)
    ann_path = project.config.annotations_dir / (fp.stem + ".json")
    if ann_path.exists():
        return Annotation.load(ann_path).model_dump()
    # no annotation yet — return an empty one sized to the image
    w, h = _image_size(fp)
    return Annotation(image=name, width=w, height=h).model_dump()


@router.put("/{project_id}/images/{name}/annotation")
def save_annotation(name: str, body: Annotation, project=Depends(get_project)) -> dict:
    fp = _frame_path(project, name)
    cfg = project.config
    cfg.annotations_dir.mkdir(parents=True, exist_ok=True)
    body.image = name
    body.reviewed = True
    body.save(cfg.annotations_dir / (fp.stem + ".json"))
    return {"saved": True, "n_boxes": len(body.boxes)}
