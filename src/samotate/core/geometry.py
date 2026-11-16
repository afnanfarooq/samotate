"""Box geometry helpers (IoU, NMS).

``iou_xyxy`` is ported verbatim from the original ``build_dataset._iou_xyxy``.
"""
from __future__ import annotations

from collections.abc import Sequence

Box = Sequence[float]  # [x1, y1, x2, y2]


def iou_xyxy(a: Box, b: Box) -> float:
    """Intersection-over-union of two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    if ua <= 0:
        return 0.0
    return inter / ua


def nms(
    boxes: list[Box],
    scores: list[float],
    iou_threshold: float,
) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices (highest score first)."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if iou_xyxy(boxes[i], boxes[j]) < iou_threshold]
    return keep
