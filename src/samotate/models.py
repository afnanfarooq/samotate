"""Data models for annotations and images.

Annotations are persisted as one JSON file per image under the project's
``annotations/`` directory. Boxes carry a ``source`` so the UI can distinguish
human-drawn boxes from SAM/auto suggestions, and a ``score`` for model boxes.
"""
from __future__ import annotations

import json
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

    def to_yolo(self, width: int, height: int) -> str:
        """Serialize to a YOLO label line: ``class cx cy w h`` (normalized)."""
        x1, y1, x2, y2 = self.xyxy
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
    def load(cls, path: str | Path) -> "Annotation":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def to_yolo_lines(self) -> list[str]:
        return [b.to_yolo(self.width, self.height) for b in self.boxes]
