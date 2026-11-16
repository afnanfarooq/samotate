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
from ..errors import SamotateError
from .extract import list_frames
from .sam import NullBackend, SAMBackend, SamNotAvailable, load_backend

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
    if not any(c.prompts for c in cfg.classes):
        # writing empty detections here would make reconcile emit an
        # all-background dataset, which looks like success but isn't
        raise SamotateError("no class has any prompts, so there is nothing to detect")

    frames = list_frames(cfg)
    if not frames:
        raise SamotateError("no frames found; run the 'extract' stage first")
    todo = [
        fp for fp in frames
        if force or not (cfg.detections_dir / (fp.stem + ".json")).exists()
    ]
    logger.info("detecting on %d of %d frames (%d cached)",
                len(todo), len(frames), len(frames) - len(todo))
    if not todo:
        return 0

    if backend is None:
        backend = load_backend()
    if isinstance(backend, NullBackend):
        raise SamNotAvailable(backend.message)

    cfg.detections_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    t0 = time.time()
    for fp in todo:
        image_rgb = _read_rgb(fp)
        if image_rgb is None:
            logger.warning("skipping unreadable image %s", fp.name)
            continue
        h, w = image_rgb.shape[:2]
        results = detect_image(cfg, image_rgb, backend)
        (cfg.detections_dir / (fp.stem + ".json")).write_text(
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
        if processed % 50 == 0:
            rate = processed / (time.time() - t0)
            logger.info("  %d/%d  %.1f img/s", processed, len(todo), rate)
    logger.info("done: %d detection files in %s", processed, cfg.detections_dir)
    return processed
