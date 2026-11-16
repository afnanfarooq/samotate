"""Stage 1: gather source images.

Either sample frames from a video (generalized from ``build_dataset.stage_extract``)
or index an existing image folder.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..config import ProjectConfig
from ..errors import SamotateError

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_frames(cfg: ProjectConfig) -> list[Path]:
    """Sample frames from ``cfg.video`` into ``cfg.frames_dir`` and return paths.

    Frames are read sequentially with ``grab()`` and only every ``stride``-th
    one is decoded. Seeking with ``CAP_PROP_POS_FRAMES`` is avoided: on many
    codecs it snaps to the nearest keyframe, which silently produces duplicate
    or misplaced frames. Reading until ``grab()`` fails also sidesteps
    ``CAP_PROP_FRAME_COUNT``, which is only an estimate for some containers.
    """
    import cv2  # imported lazily so the package imports without OpenCV present

    if not Path(cfg.video).is_file():
        raise SamotateError(f"video not found: {cfg.video}")
    cap = cv2.VideoCapture(str(cfg.video))
    if not cap.isOpened():
        raise SamotateError(f"could not open video (unsupported codec?): {cfg.video}")

    frames_dir = cfg.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # estimate; logging only
    stride = max(1, int(round(src_fps / cfg.fps)))
    logger.info(
        "src fps=%.2f ~%d frames, stride=%d (~%d to sample%s)",
        src_fps, n_total, stride, -(-n_total // stride),
        f", capped at {cfg.limit}" if cfg.limit else "",
    )

    saved: list[Path] = []
    idx = 0
    try:
        while not (cfg.limit and len(saved) >= cfg.limit):
            if not cap.grab():
                break
            if idx % stride == 0:
                ok, frame = cap.retrieve()
                if ok:
                    out_path = frames_dir / f"frame_{idx:07d}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved.append(out_path)
                    if len(saved) % 200 == 0:
                        logger.info("  %d frames extracted", len(saved))
            idx += 1
    finally:
        cap.release()

    if not saved:
        raise SamotateError(f"no frames could be read from {cfg.video}")
    logger.info("done: %d frames in %s", len(saved), frames_dir)
    return saved


def index_images(cfg: ProjectConfig) -> list[Path]:
    """Copy an existing image folder into ``frames_dir`` and return paths.

    Images are brought under ``frames_dir`` so the rest of the pipeline can treat
    video and folder sources identically.
    """
    src_dir = Path(cfg.images_dir)
    if not src_dir.is_dir():
        raise SamotateError(f"images_dir not found: {src_dir}")
    sources = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not sources:
        raise SamotateError(
            f"no images in {src_dir} (looked for {', '.join(sorted(IMAGE_EXTS))})"
        )
    if cfg.limit:
        sources = sources[: cfg.limit]

    # annotations, labels and detections are keyed by file stem, so two files
    # like a.jpg and a.png would silently share one label file
    stems: dict[str, Path] = {}
    for p in sources:
        if p.stem in stems:
            raise SamotateError(
                f"{stems[p.stem].name} and {p.name} share a file stem; rename one"
            )
        stems[p.stem] = p

    frames_dir = cfg.frames_dir
    frames_dir.mkdir(parents=True, exist_ok=True)
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
