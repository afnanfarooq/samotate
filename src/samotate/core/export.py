"""Stage 4: write YOLO labels, a temporal train/val split, and data.yaml.

Generalized from ``build_dataset.stage_reconcile`` (label writing) and
``stage_split``. Class ``names:`` come from the config. The split supports a
Windows-safe ``copy`` mode in addition to the original symlink behavior.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from ..config import ProjectConfig
from ..models import Annotation
from .extract import list_frames

logger = logging.getLogger(__name__)


def write_yolo_labels(cfg: ProjectConfig) -> int:
    """Write one ``.txt`` per annotation into ``labels/``. Returns count."""
    cfg.labels_dir.mkdir(parents=True, exist_ok=True)
    ann_files = sorted(cfg.annotations_dir.glob("*.json"))
    for af in ann_files:
        ann = Annotation.load(af)
        lines = ann.to_yolo_lines()
        text = ("\n".join(lines) + "\n") if lines else ""
        (cfg.labels_dir / (af.stem + ".txt")).write_text(text, encoding="utf-8")
    logger.info("wrote %d YOLO label files", len(ann_files))
    return len(ann_files)


def _place(src: Path, dest: Path, mode: str) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if mode == "symlink":
        dest.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dest)


def split_dataset(cfg: ProjectConfig, every: int = 1, offset: int = 0) -> dict:
    """Temporal train/val split into ``images/{train,val}`` + ``labels/{train,val}``."""
    img_train = cfg.root / "images" / "train"
    img_val = cfg.root / "images" / "val"
    lbl_train = cfg.root / "labels" / "train"
    lbl_val = cfg.root / "labels" / "val"
    for d in (img_train, img_val, lbl_train, lbl_val):
        d.mkdir(parents=True, exist_ok=True)
        for p in d.iterdir():
            if p.is_symlink() or p.is_file():
                p.unlink()

    frames = list_frames(cfg)
    if every > 1:
        frames = frames[offset::every]
    cut = int(len(frames) * cfg.train_fraction)
    train, val = frames[:cut], frames[cut:]
    logger.info("split: %d train / %d val", len(train), len(val))

    def place_set(fset, img_dir, lbl_dir):
        placed = 0
        for fp in fset:
            lbl_src = cfg.labels_dir / (fp.stem + ".txt")
            if not lbl_src.exists():
                continue
            _place(fp, img_dir / fp.name, cfg.link_mode)
            _place(lbl_src, lbl_dir / lbl_src.name, cfg.link_mode)
            placed += 1
        return placed

    n_train = place_set(train, img_train, lbl_train)
    n_val = place_set(val, img_val, lbl_val)

    names = "\n".join(f"  {c.id}: {c.name}" for c in sorted(cfg.classes, key=lambda c: c.id))
    yaml = (
        f"# Ultralytics YOLO dataset ({cfg.name})\n"
        f"path: {cfg.root.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n{names}\n"
    )
    data_yaml = cfg.root / "data.yaml"
    data_yaml.write_text(yaml, encoding="utf-8")
    logger.info("wrote %s", data_yaml)
    return {"train": n_train, "val": n_val, "data_yaml": str(data_yaml)}


def export_dataset(cfg: ProjectConfig, every: int = 1, offset: int = 0) -> dict:
    """Write labels then split — the full export step."""
    write_yolo_labels(cfg)
    return split_dataset(cfg, every=every, offset=offset)
