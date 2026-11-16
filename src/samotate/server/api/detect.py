"""Interactive SAM assist: text prompt or click point on a single image."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ...core.sam import SamNotAvailable
from .deps import frame_path, get_project, get_state

router = APIRouter(prefix="/api/projects", tags=["detect"])


# Optional[...] rather than `X | None`: Pydantic evaluates these annotations at
# runtime, and the `|` form fails on Python 3.9.
class InteractiveRequest(BaseModel):
    image: str
    prompt: Optional[str] = None
    threshold: float = Field(0.5, ge=0.0, le=1.0)
    point: Optional[tuple[float, float]] = None


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
    if body.point is None and not body.prompt:
        raise HTTPException(400, "provide 'prompt' or 'point'")
    fp = frame_path(project.config, body.image)
    image_rgb = _read_rgb(fp)
    backend = get_state(request).backend()
    try:
        if body.point is not None:
            boxes = backend.point_prompt(image_rgb, body.point[0], body.point[1])
        else:
            boxes = backend.text_prompt(image_rgb, body.prompt, body.threshold)
    except SamNotAvailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"boxes": boxes}
