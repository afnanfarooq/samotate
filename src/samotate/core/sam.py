"""SAM backend abstraction.

The original script imported SAM 3.1 directly and forced CUDA bfloat16 autocast.
Here the model sits behind a small ``SAMBackend`` protocol so:

  * the rest of the app never imports torch/sam directly;
  * the tool is fully usable for *manual* annotation when SAM isn't installed
    (``NullBackend`` raises a clear, actionable error only when assist is used);
  * device/precision are detected rather than hardcoded to CUDA.

SAM 3.1 is a research package and is not on PyPI — see the README for setup.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


class SamNotAvailable(RuntimeError):
    """Raised when a SAM-assist feature is used but no backend is installed."""


@runtime_checkable
class SAMBackend(Protocol):
    """Minimal interface the app needs from a segmentation/detection model."""

    def text_prompt(
        self, image_rgb: np.ndarray, prompt: str, threshold: float
    ) -> list[dict]:
        """Return ``[{"bbox_xyxy": [x1,y1,x2,y2], "score": float}, ...]``."""
        ...

    def point_prompt(
        self, image_rgb: np.ndarray, x: float, y: float
    ) -> list[dict]:
        """Return candidate boxes for a single foreground click."""
        ...


class NullBackend:
    """Placeholder backend used when SAM is not installed.

    Manual annotation works without SAM; any assist call fails loudly with
    install guidance instead of silently doing nothing.
    """

    _MSG = (
        "SAM is not installed. Install the optional model backend to use "
        "auto-detection / prompt assist (see the README 'SAM setup' section). "
        "Manual annotation works without it."
    )

    def text_prompt(self, image_rgb, prompt, threshold):  # noqa: D102, ANN001
        raise SamNotAvailable(self._MSG)

    def point_prompt(self, image_rgb, x, y):  # noqa: D102, ANN001
        raise SamNotAvailable(self._MSG)


class Sam3Backend:
    """Real SAM 3.1 backend, wrapping the calls used by the original script.

    Construction is lazy and tolerant: importing torch/sam3 happens here, not at
    module import, so the package and its tests load without a GPU.
    """

    def __init__(self, confidence_threshold: float = 0.4) -> None:
        import torch  # noqa: PLC0415
        from sam3 import build_sam3_image_model  # noqa: PLC0415
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415

        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            else "cpu"
        )
        self._device = device
        if device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        logger.info("loading SAM 3.1 on %s", device)
        model = build_sam3_image_model()
        self._processor = Sam3Processor(
            model, confidence_threshold=confidence_threshold
        )

    def _to_pil(self, image_rgb: np.ndarray):
        from PIL import Image  # noqa: PLC0415

        return Image.fromarray(image_rgb)

    def text_prompt(self, image_rgb, prompt, threshold):
        state = self._processor.set_image(self._to_pil(image_rgb))
        self._processor.reset_all_prompts(state)
        out = self._processor.set_text_prompt(state=state, prompt=prompt)
        boxes = out["boxes"].float().cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        keep = scores >= threshold
        return [
            {"bbox_xyxy": b.tolist(), "score": float(s)}
            for b, s in zip(boxes[keep], scores[keep])
        ]

    def point_prompt(self, image_rgb, x, y):
        state = self._processor.set_image(self._to_pil(image_rgb))
        self._processor.reset_all_prompts(state)
        out = self._processor.set_point_prompt(
            state=state, points=[[x, y]], labels=[1]
        )
        boxes = out["boxes"].float().cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        return [
            {"bbox_xyxy": b.tolist(), "score": float(s)}
            for b, s in zip(boxes, scores)
        ]


def load_backend(confidence_threshold: float = 0.4) -> SAMBackend:
    """Return a real SAM backend if available, else a :class:`NullBackend`."""
    try:
        return Sam3Backend(confidence_threshold=confidence_threshold)
    except Exception as exc:  # noqa: BLE001 — any import/load failure → manual mode
        logger.warning("SAM backend unavailable (%s); manual mode only", exc)
        return NullBackend()
