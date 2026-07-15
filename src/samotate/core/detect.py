"""Stage 2: run SAM prompts over frames and cache raw detections.

Generalized from ``build_dataset.stage_detect`` / ``_detect_one``: instead of
hardcoded staff/person prompts it iterates over every prompt defined on every
class in the config. The cached JSON keeps a per-class shape so reconciliation
can map detections back to classes.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from ..config import ProjectConfig
from .extract import list_frames
from .sam import SAMBackend, load_backend

logger = logging.getLogger(__name__)


def _read_rgb(path: Path):
    import cv2  # noqa: PLC0415

    frame = cv2.imread(str(path))
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_image(cfg: ProjectConfig, image_rgb, backend: SAMBackend) -> dict:
    """Run every class prompt on one image; return ``{class_name: [boxes]}``."""
    results: dict[str, list[dict]] = {}
    for cls in cfg.classes:
        hits: list[dict] = []
        for prompt in cls.prompts:
            hits.extend(
                backend.text_prompt(image_rgb, prompt.text, prompt.threshold)
            )
        results[cls.name] = hits
    return results


def detect_all(
    cfg: ProjectConfig,
    backend: Optional[SAMBackend] = None,
    force: bool = False,
) -> int:
    """Detect over all frames, caching one JSON per frame. Returns count processed."""
    cfg.detections_dir.mkdir(parents=True, exist_ok=True)
    frames = list_frames(cfg)
    logger.info("detecting on %d frames", len(frames))
    if backend is None:
        backend = load_backend()

    processed = 0
    t0 = time.time()
    for i, fp in enumerate(frames):
        out_path = cfg.detections_dir / (fp.stem + ".json")
        if out_path.exists() and not force:
            continue
        image_rgb = _read_rgb(fp)
        if image_rgb is None:
            continue
        h, w = image_rgb.shape[:2]
        results = detect_image(cfg, image_rgb, backend)
        out_path.write_text(
            json.dumps(
                {
                    "frame": fp.name,
                    "width": int(w),
                    "height": int(h),
                    "detections": results,
                }
            ),
            encoding="utf-8",
        )
        processed += 1
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - t0)
            logger.info("  %d/%d  %.1f fps", i + 1, len(frames), rate)
    logger.info("done: %d detection files in %s", processed, cfg.detections_dir)
    return processed
