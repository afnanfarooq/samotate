"""SAM backend abstraction.

The model sits behind a small ``SAMBackend`` protocol so:

  * the rest of the app never imports torch/sam directly;
  * extract, reconcile, export and preview work when SAM isn't installed
    (``NullBackend`` raises a clear, actionable error only when detection runs);
  * the device is detected (CUDA > MPS > CPU) rather than hardcoded, and can be
    forced with ``SAMOTATE_DEVICE``.

SAM 3.1 is a research package and is not on PyPI — see the README for setup.
The calls below follow ``sam3.model.sam3_image_processor.Sam3Processor`` and
``Sam3Image.predict_inst`` from https://github.com/facebookresearch/sam3.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)

DEVICE_ENV = "SAMOTATE_DEVICE"
_DEVICES = ("cuda", "mps", "cpu")


class SamNotAvailable(RuntimeError):
    """Raised when a SAM feature is used but no backend could be loaded."""


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
    """Placeholder backend used when SAM could not be loaded.

    Extract, reconcile, export and preview work without SAM; any detection call
    fails loudly with the load error and install guidance instead of silently
    doing nothing.
    """

    def __init__(self, reason: str = "not installed") -> None:
        self.reason = reason

    @property
    def message(self) -> str:
        return (
            f"SAM backend unavailable ({self.reason}). Install it to use "
            "detection (see the README 'SAM setup' section)."
        )

    def text_prompt(self, image_rgb, prompt, threshold):  # noqa: D102, ANN001
        raise SamNotAvailable(self.message)

    def point_prompt(self, image_rgb, x, y):  # noqa: D102, ANN001
        raise SamNotAvailable(self.message)


def pick_device(torch: Any, requested: Optional[str] = None) -> str:
    """Choose ``cuda``, ``mps`` or ``cpu``.

    ``requested`` (or the ``SAMOTATE_DEVICE`` env var) forces a device; it is
    an error to force one that isn't available rather than silently falling
    back to a much slower one.
    """
    requested = (requested or os.environ.get(DEVICE_ENV) or "").strip().lower()
    mps = getattr(torch.backends, "mps", None)
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(mps and mps.is_available()),
        "cpu": True,
    }
    if requested:
        if requested not in _DEVICES:
            raise ValueError(f"{DEVICE_ENV} must be one of {', '.join(_DEVICES)}")
        if not available[requested]:
            raise RuntimeError(f"device '{requested}' requested but not available")
        return requested
    return next(d for d in _DEVICES if available[d])


def _to_numpy(t: Any) -> np.ndarray:
    """Tensor (any device/dtype, incl. bfloat16) or array → float numpy array."""
    if hasattr(t, "detach"):
        return t.detach().float().cpu().numpy()
    return np.asarray(t, dtype=np.float32)


def mask_to_xyxy(mask: np.ndarray) -> Optional[list[float]]:
    """Tight ``[x1, y1, x2, y2]`` box around a binary mask, or None if empty."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


class Sam3Backend:
    """SAM 3 / 3.1 image backend.

    Construction imports torch/sam3 lazily, so the package and its tests load
    without either installed.

    The image encoder is by far the most expensive step, so the encoded image
    is cached: consecutive calls with the *same array object* (as
    ``detect_image`` makes, once per prompt) reuse it and only re-run the
    prompt heads. Don't mutate an array in place between calls.
    """

    def __init__(self, *, points: bool = False, device: Optional[str] = None) -> None:
        import torch  # noqa: PLC0415
        from sam3 import build_sam3_image_model  # noqa: PLC0415
        from sam3.model.sam3_image_processor import Sam3Processor  # noqa: PLC0415

        self._torch = torch
        self.device = pick_device(torch, device)
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.device == "mps":
            logger.warning("SAM 3 support for MPS is preliminary upstream; results may "
                           "differ from CUDA. Set %s=cpu to compare.", DEVICE_ENV)
        logger.info("loading SAM 3 on %s%s", self.device,
                    " (with point prompts)" if points else "")

        # the builder only knows "cuda" (calls .cuda()) or leaves the model on CPU
        model = build_sam3_image_model(
            device="cuda" if self.device == "cuda" else "cpu",
            enable_inst_interactivity=points,
        )
        if self.device == "mps":
            model = model.to("mps")
        self._model = model
        self._points = points
        # the processor defaults to device="cuda"; it must match the model
        self._processor = Sam3Processor(model, device=self.device)

        self._lock = threading.Lock()  # the cached state is not shareable
        self._image: Optional[np.ndarray] = None
        self._state: Optional[dict] = None

    def _autocast(self):
        # upstream runs CUDA inference under bfloat16 autocast
        if self.device == "cuda":
            return self._torch.autocast("cuda", dtype=self._torch.bfloat16)
        return contextlib.nullcontext()

    def _state_for(self, image_rgb: np.ndarray) -> dict:
        if image_rgb is not self._image:
            from PIL import Image  # noqa: PLC0415

            # PIL, not ndarray: set_image reads (h, w) from shape[-2:], which is
            # wrong for an HWC array
            self._state = self._processor.set_image(Image.fromarray(image_rgb))
            self._image = image_rgb  # holding the ref keeps the identity check valid
        return self._state

    def text_prompt(self, image_rgb, prompt, threshold):
        with self._lock, self._autocast():
            state = self._state_for(image_rgb)
            self._processor.reset_all_prompts(state)  # keeps the image features
            # filtering happens inside the processor, so apply this prompt's
            # threshold there rather than a fixed one chosen at load time
            self._processor.set_confidence_threshold(threshold)
            out = self._processor.set_text_prompt(prompt=prompt, state=state)
            boxes = _to_numpy(out["boxes"]).reshape(-1, 4)
            scores = _to_numpy(out["scores"]).reshape(-1)
        return [
            {"bbox_xyxy": b.tolist(), "score": float(s)}
            for b, s in zip(boxes, scores)
            if s >= threshold
        ]

    def point_prompt(self, image_rgb, x, y):
        if not self._points:
            raise SamNotAvailable("point prompts need the backend loaded with points=True")
        with self._lock, self._autocast(), self._torch.inference_mode():
            state = self._state_for(image_rgb)
            masks, scores, _ = self._model.predict_inst(
                state,
                point_coords=np.array([[x, y]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                multimask_output=True,
            )
        masks, scores = np.asarray(masks), _to_numpy(scores).reshape(-1)
        out = []
        for i in np.argsort(scores)[::-1]:
            box = mask_to_xyxy(masks[i] > 0)
            if box is not None:
                out.append({"bbox_xyxy": box, "score": float(scores[i])})
        return out


def load_backend(*, points: bool = False) -> SAMBackend:
    """Return a real SAM backend if available, else a :class:`NullBackend`.

    ``points=True`` also loads the interactive (click) head, which costs extra
    memory; batch detection only needs text prompts.
    """
    try:
        return Sam3Backend(points=points)
    except ImportError as exc:
        logger.warning("SAM backend not installed (%s)", exc)
        return NullBackend(f"not installed: {exc}")
    except Exception as exc:  # noqa: BLE001 — installed but failed to load
        # e.g. missing checkpoint/HF access or out of memory: keep the
        # traceback visible so this isn't mistaken for "not installed"
        logger.error("SAM backend failed to load", exc_info=True)
        return NullBackend(f"failed to load: {exc}")
