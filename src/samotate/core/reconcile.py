"""Stage 3: turn cached detections into final per-image annotations.

Two strategies (selected via ``cfg.reconcile.strategy``):

  * ``per_prompt`` — every detection becomes a box of its class; overlapping
    boxes of the same class are de-duplicated by NMS. Works for any task.
  * ``hierarchical`` — reproduces the original staff/customer logic: a broad
    ``base_class`` is subdivided by more specific classes via IoU matching, with
    optional ``override_classes`` that win ties (e.g. "seated diner" → customer).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..config import ProjectConfig
from ..models import Annotation, Box
from .geometry import iou_xyxy, nms

logger = logging.getLogger(__name__)


def _load_detection(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reconcile_per_prompt(cfg: ProjectConfig, meta: dict) -> list[Box]:
    boxes: list[Box] = []
    for cls in cfg.classes:
        dets = meta["detections"].get(cls.name, [])
        if not dets:
            continue
        bxs = [d["bbox_xyxy"] for d in dets]
        scores = [d.get("score", 1.0) for d in dets]
        for i in nms(bxs, scores, cfg.reconcile.nms_iou):
            boxes.append(
                Box(
                    class_id=cls.id,
                    xyxy=tuple(bxs[i]),
                    source="auto",
                    score=scores[i],
                )
            )
    return boxes


def _reconcile_hierarchical(cfg: ProjectConfig, meta: dict) -> list[Box]:
    rc = cfg.reconcile
    base = cfg.class_by_name(rc.base_class)
    overrides = [cfg.class_by_name(n) for n in rc.override_classes]
    specific = [
        c for c in cfg.classes
        if c.name != base.name and c.name not in rc.override_classes
    ]

    base_dets = meta["detections"].get(base.name, [])
    override_dets = [
        d for c in overrides for d in meta["detections"].get(c.name, [])
    ]
    # flatten specific-class detections, remembering which class each came from
    specific_dets = [
        {**d, "_cls": c.id}
        for c in specific
        for d in meta["detections"].get(c.name, [])
    ]

    boxes: list[Box] = []
    used_specific: set[int] = set()
    # a base box that matches no specific class keeps the base class
    # (mirrors the original: an unmatched "person" stays "customer")
    residual_id = base.id

    for bd in base_dets:
        bb = bd["bbox_xyxy"]
        best_iou, best_i = 0.0, -1
        for i, sd in enumerate(specific_dets):
            if i in used_specific:
                continue
            iou = iou_xyxy(bb, sd["bbox_xyxy"])
            if iou > best_iou:
                best_iou, best_i = iou, i
        matched = best_iou >= rc.iou_threshold
        if matched:
            ov_iou = max(
                (iou_xyxy(bb, od["bbox_xyxy"]) for od in override_dets),
                default=0.0,
            )
            if ov_iou >= rc.override_iou:
                cls_id = residual_id
            else:
                cls_id = specific_dets[best_i]["_cls"]
                used_specific.add(best_i)
        else:
            cls_id = residual_id
        boxes.append(
            Box(class_id=cls_id, xyxy=tuple(bb), source="auto",
                score=bd.get("score"))
        )

    # orphan specific boxes: emit unless an override agreed
    for i, sd in enumerate(specific_dets):
        if i in used_specific:
            continue
        ov_iou = max(
            (iou_xyxy(sd["bbox_xyxy"], od["bbox_xyxy"]) for od in override_dets),
            default=0.0,
        )
        if ov_iou >= rc.override_iou:
            continue
        boxes.append(
            Box(class_id=sd["_cls"], xyxy=tuple(sd["bbox_xyxy"]),
                source="auto", score=sd.get("score"))
        )
    return boxes


def reconcile_meta(cfg: ProjectConfig, meta: dict) -> Annotation:
    """Reconcile one detection dict into an :class:`Annotation`."""
    if cfg.reconcile.strategy == "hierarchical":
        boxes = _reconcile_hierarchical(cfg, meta)
    else:
        boxes = _reconcile_per_prompt(cfg, meta)
    return Annotation(
        image=meta["frame"], width=meta["width"], height=meta["height"],
        boxes=boxes,
    )


def reconcile_all(cfg: ProjectConfig) -> int:
    """Reconcile every cached detection into an annotation JSON. Returns count."""
    cfg.annotations_dir.mkdir(parents=True, exist_ok=True)
    det_files = sorted(cfg.detections_dir.glob("*.json"))
    logger.info("reconciling %d frames (strategy=%s)",
                len(det_files), cfg.reconcile.strategy)
    n = 0
    for df in det_files:
        ann = reconcile_meta(cfg, _load_detection(df))
        out = cfg.annotations_dir / (df.stem + ".json")
        # don't clobber human edits if the image was already reviewed
        if out.exists():
            existing = Annotation.load(out)
            if existing.reviewed:
                continue
        ann.save(out)
        n += 1
    logger.info("done: wrote %d annotation files", n)
    return n
