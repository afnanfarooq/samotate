"""Configuration models for a samotate project.

This replaces all the hardcoded constants in the original ``build_dataset``
script (paths, prompts, thresholds) with a validated, YAML-backed config so the
tool works for arbitrary classes and detection tasks rather than a single
hardcoded retail video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Prompt(BaseModel):
    """A single SAM text prompt and the score threshold to keep its boxes."""

    text: str
    threshold: float = 0.5

    @field_validator("threshold")
    @classmethod
    def _valid_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return v


class ClassDef(BaseModel):
    """A label class: an id, a display name, and zero or more SAM prompts.

    Prompts are optional — a class with no prompts is still fully usable for
    manual annotation; it simply gets no SAM auto-suggestions.
    """

    id: int
    name: str
    prompts: list[Prompt] = Field(default_factory=list)
    color: Optional[str] = None  # hex like "#ff0044"; used by `preview`, auto-picked if omitted


class ReconcileConfig(BaseModel):
    """How overlapping detections are resolved into final boxes.

    - ``per_prompt`` (default): every prompt hit becomes a box of its class;
      overlapping boxes of the *same* class are de-duplicated by NMS.
    - ``hierarchical``: reproduces the original staff/customer behavior — a
      broad ``base_class`` is subdivided by more specific classes via IoU
      matching, with optional override classes that win ties.
    """

    strategy: Literal["per_prompt", "hierarchical"] = "per_prompt"
    iou_threshold: float = Field(0.4, ge=0.0, le=1.0)
    nms_iou: float = Field(0.5, ge=0.0, le=1.0)
    # hierarchical-only knobs
    base_class: Optional[str] = None
    override_classes: list[str] = Field(default_factory=list)
    override_iou: float = Field(0.5, ge=0.0, le=1.0)


class ProjectConfig(BaseModel):
    """Top-level project configuration loaded from ``config.yaml``."""

    name: str = "samotate-project"
    # --- source: exactly one of video / images_dir ---
    video: Optional[Path] = None
    images_dir: Optional[Path] = None
    # --- output / working dir ---
    root: Path = Path("./dataset")
    # --- frame sampling (video only) ---
    fps: float = Field(1.0, gt=0.0)
    limit: Optional[int] = Field(None, gt=0)
    # --- classes & detection ---
    classes: list[ClassDef]
    reconcile: ReconcileConfig = Field(default_factory=ReconcileConfig)
    # --- export ---
    train_fraction: float = Field(0.8, gt=0.0, le=1.0)
    link_mode: Literal["copy", "symlink"] = "copy"

    @model_validator(mode="after")
    def _check_source_and_classes(self) -> ProjectConfig:
        if self.video is None and self.images_dir is None:
            raise ValueError("config must set either 'video' or 'images_dir'")
        if self.video is not None and self.images_dir is not None:
            raise ValueError("set only one of 'video' or 'images_dir', not both")
        if not self.classes:
            raise ValueError("config must define at least one class")
        ids = [c.id for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("class ids must be unique")
        # YOLO derives the class count from names: an id outside 0..N-1 makes
        # Ultralytics reject the label ("exceeds dataset class count").
        if sorted(ids) != list(range(len(ids))):
            raise ValueError(
                f"class ids must be 0..{len(ids) - 1} with no gaps (got {sorted(ids)})"
            )
        names = [c.name for c in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("class names must be unique")
        if self.reconcile.strategy == "hierarchical":
            if self.reconcile.base_class is None:
                raise ValueError("hierarchical reconcile requires 'base_class'")
            if self.reconcile.base_class not in names:
                raise ValueError("reconcile.base_class must name an existing class")
            for oc in self.reconcile.override_classes:
                if oc not in names:
                    raise ValueError(f"override class '{oc}' is not a defined class")
        return self

    # --- derived output paths (mirror the original build_dataset layout) ---
    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def detections_dir(self) -> Path:
        return self.root / "detections"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels"

    def class_by_name(self, name: str) -> ClassDef:
        for c in self.classes:
            if c.name == name:
                return c
        raise KeyError(name)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
        """Load a config file.

        Relative ``video``, ``images_dir`` and ``root`` paths are resolved
        against the config file's directory, not the current working
        directory, so ``samotate pipeline -c some/dir/config.yaml`` behaves the
        same wherever it is run from.
        """
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        base = path.resolve().parent
        data.setdefault("root", "./dataset")
        for key in ("video", "images_dir", "root"):
            if data.get(key):
                p = Path(data[key]).expanduser()
                data[key] = str(p if p.is_absolute() else base / p)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        data = self.model_dump(mode="json", exclude_none=True)
        Path(path).write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
