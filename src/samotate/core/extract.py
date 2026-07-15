"""Stage 1: gather source images.

Either sample frames from a video (generalized from ``build_dataset.stage_extract``)
or index an existing image folder. The cv2 frame-stride math is kept as-is.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from ..config import ProjectConfig

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_frames(cfg: ProjectConfig) -> list[Path]:
    """Sample frames from ``cfg.video`` into ``cfg.frames_dir`` and return paths."""
    import cv2  # imported lazily so the package imports without OpenCV present

    frames_dir = cfg.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(cfg.video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(src_fps / cfg.fps)))
    indices = list(range(0, n_total, stride))
    if cfg.limit:
        indices = indices[: cfg.limit]
    logger.info(
        "src fps=%.2f total=%d stride=%d sampling %d frames",
        src_fps, n_total, stride, len(indices),
    )

    saved: list[Path] = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = frames_dir / f"frame_{idx:07d}.jpg"
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(out_path)
        if (i + 1) % 200 == 0:
            logger.info("  %d/%d frames extracted", i + 1, len(indices))
    cap.release()
    logger.info("done: %d frames in %s", len(saved), frames_dir)
    return saved


def index_images(cfg: ProjectConfig) -> list[Path]:
    """Copy/symlink an existing image folder into ``frames_dir`` and return paths.

    Images are brought under ``frames_dir`` so the rest of the pipeline can treat
    video and folder sources identically.
    """
    frames_dir = cfg.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(cfg.images_dir)
    sources = sorted(
        p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if cfg.limit:
        sources = sources[: cfg.limit]

    out: list[Path] = []
    for p in sources:
        dest = frames_dir / p.name
        if not dest.exists():
            shutil.copy2(p, dest)
        out.append(dest)
    logger.info("indexed %d images into %s", len(out), frames_dir)
    return out


def gather_source_images(cfg: ProjectConfig) -> list[Path]:
    """Resolve the configured source into a sorted list of image paths."""
    if cfg.video is not None:
        return extract_frames(cfg)
    return index_images(cfg)


def list_frames(cfg: ProjectConfig) -> list[Path]:
    """List already-extracted frames without re-running extraction."""
    if not cfg.frames_dir.exists():
        return []
    return sorted(
        p for p in cfg.frames_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
