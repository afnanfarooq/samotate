"""Interactive SAM assist: text prompt or click point on a single image."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ...core.sam import SamNotAvailable
from .annotations import _frame_path
from .deps import get_project, get_state

router = APIRouter(prefix="/api/projects", tags=["detect"])


class InteractiveRequest(BaseModel):
    image: str
    prompt: str | None = None
    threshold: float = 0.5
    point: tuple[float, float] | None = None


def _read_rgb(fp):
    import cv2

    frame = cv2.imread(str(fp))
    if frame is None:
        raise HTTPException(400, "could not read image")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@router.post("/{project_id}/detect-interactive")
def detect_interactive(
    request: Request, body: InteractiveRequest, project=Depends(get_project)
) -> dict:
    fp = _frame_path(project, body.image)
    backend = get_state(request).backend()
    image_rgb = _read_rgb(fp)
    try:
        if body.point is not None:
            boxes = backend.point_prompt(image_rgb, body.point[0], body.point[1])
        elif body.prompt:
            boxes = backend.text_prompt(image_rgb, body.prompt, body.threshold)
        else:
            raise HTTPException(400, "provide 'prompt' or 'point'")
    except SamNotAvailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"boxes": boxes}
