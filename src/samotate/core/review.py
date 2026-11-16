"""Headless inspection: project status and box previews.

With no UI, these are how you check auto-labels: ``status`` reports progress
per stage, and ``preview`` draws the boxes onto copies of the images so you can
flip through them in any image viewer.
"""
from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

from ..config import ProjectConfig
from ..errors import SamotateError
from ..models import Annotation
from .extract import list_frames

logger = logging.getLogger(__name__)

# BGR, used for classes without an explicit color
_PALETTE = [
    (48, 59, 255), (89, 199, 52), (255, 132, 10), (10, 214, 255),
    (222, 82, 175), (199, 199, 48), (0, 149, 255), (147, 112, 219),
]


def project_status(cfg: ProjectConfig) -> dict:
    """Counts per pipeline stage plus boxes per class."""
    frames = list_frames(cfg)
    anns = [Annotation.load(p) for p in sorted(cfg.annotations_dir.glob("*.json"))]
    per_class: Counter[int] = Counter(b.class_id for a in anns for b in a.boxes)
    return {
        "frames": len(frames),
        "detections": len(list(cfg.detections_dir.glob("*.json"))),
        "annotations": len(anns),
        "reviewed": sum(a.reviewed for a in anns),
        "empty": sum(not a.boxes for a in anns),
        "labels": len(list(cfg.labels_dir.glob("*.txt"))),
        "boxes": {c.name: per_class.get(c.id, 0) for c in cfg.classes},
    }


def _bgr(hex_color: Optional[str], class_id: int) -> tuple[int, int, int]:
    if hex_color:
        h = hex_color.lstrip("#")
        if len(h) == 6:
            try:
                r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
                return (b, g, r)
            except ValueError:
                pass
    return _PALETTE[class_id % len(_PALETTE)]


def render_previews(
    cfg: ProjectConfig,
    out_dir: Optional[Path] = None,
    limit: Optional[int] = None,
) -> Path:
    """Draw each annotation's boxes onto a copy of its image. Returns ``out_dir``."""
    import cv2  # noqa: PLC0415

    ann_files = sorted(cfg.annotations_dir.glob("*.json"))
    if not ann_files:
        raise SamotateError("no annotations to preview; run the 'reconcile' stage first")
    if limit:
        ann_files = ann_files[:limit]
    out_dir = Path(out_dir) if out_dir else cfg.root / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = {c.id: c for c in cfg.classes}
    written = 0
    for af in ann_files:
        ann = Annotation.load(af)
        img = cv2.imread(str(cfg.frames_dir / ann.image))
        if img is None:
            logger.warning("skipping %s: image not found or unreadable", ann.image)
            continue
        thick = max(2, round(max(img.shape[:2]) / 400))
        for box in ann.boxes:
            cls = classes.get(box.class_id)
            name = cls.name if cls else f"class {box.class_id}"
            color = _bgr(cls.color if cls else None, box.class_id)
            x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
            label = name if box.score is None else f"{name} {box.score:.2f}"
            scale = thick / 3
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            ty = max(y1, th + base + 2)
            cv2.rectangle(img, (x1, ty - th - base - 2), (x1 + tw + 4, ty), color, -1)
            cv2.putText(img, label, (x1 + 2, ty - base), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / (af.stem + ".jpg")), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        written += 1
    logger.info("wrote %d previews to %s", written, out_dir)
    return out_dir
