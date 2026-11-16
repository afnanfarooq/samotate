"""Data models for annotations and images.

Annotations are persisted as one JSON file per image under the project's
``annotations/`` directory. Boxes carry a ``source`` to distinguish
human-drawn boxes from SAM/auto suggestions, and a ``score`` for model boxes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

BoxSource = Literal["manual", "sam", "auto"]


class Box(BaseModel):
    """An axis-aligned bounding box in absolute pixel coordinates (xyxy)."""

    class_id: int
    xyxy: tuple[float, float, float, float]
    source: BoxSource = "manual"
    score: Optional[float] = None

    def to_yolo(self, width: int, height: int) -> Optional[str]:
        """Serialize to a YOLO label line: ``class cx cy w h`` (normalized).

        The box is clipped to the image first: model boxes often overshoot the
        border by a pixel, and Ultralytics discards an image's labels entirely
        if any coordinate falls outside [0, 1]. Returns ``None`` for a box with
        no area left after clipping.
        """
        def clip(v: float, hi: int) -> float:
            return min(max(v, 0.0), float(hi))

        x1, x2 = sorted((clip(self.xyxy[0], width), clip(self.xyxy[2], width)))
        y1, y2 = sorted((clip(self.xyxy[1], height), clip(self.xyxy[3], height)))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        return f"{self.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


class Annotation(BaseModel):
    """All boxes for a single image, plus its pixel dimensions."""

    image: str
    width: int
    height: int
    boxes: list[Box] = Field(default_factory=list)
    # set once a human has reviewed/saved this image
    reviewed: bool = False

    @classmethod
    def load(cls, path: str | Path) -> Annotation:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def to_yolo_lines(self) -> list[str]:
        lines = (b.to_yolo(self.width, self.height) for b in self.boxes)
        return [line for line in lines if line is not None]
