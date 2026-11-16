"""Get and save per-image annotations."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ...models import Annotation
from .deps import frame_path, get_project

router = APIRouter(prefix="/api/projects", tags=["annotations"])


def _image_size(fp):
    from PIL import Image

    with Image.open(fp) as im:
        return im.width, im.height


@router.get("/{project_id}/images/{name}/annotation")
def get_annotation(name: str, project=Depends(get_project)) -> dict:
    cfg = project.config
    fp = frame_path(cfg, name)
    ann_path = cfg.annotations_dir / (fp.stem + ".json")
    if ann_path.exists():
        return Annotation.load(ann_path).model_dump()
    # no annotation yet — return an empty one sized to the image
    w, h = _image_size(fp)
    return Annotation(image=name, width=w, height=h).model_dump()


@router.put("/{project_id}/images/{name}/annotation")
def save_annotation(name: str, body: Annotation, project=Depends(get_project)) -> dict:
    cfg = project.config
    fp = frame_path(cfg, name)
    known = {c.id for c in cfg.classes}
    unknown = sorted({b.class_id for b in body.boxes} - known)
    if unknown:
        raise HTTPException(400, f"unknown class id(s): {unknown}")
    # trust the file, not the client, for dimensions: YOLO export normalizes by them
    body.width, body.height = _image_size(fp)
    body.image = name
    body.reviewed = True
    cfg.annotations_dir.mkdir(parents=True, exist_ok=True)
    body.save(cfg.annotations_dir / (fp.stem + ".json"))
    return {"saved": True, "n_boxes": len(body.boxes)}
