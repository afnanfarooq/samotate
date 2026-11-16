"""Sam3Backend logic, tested with fakes that mimic the sam3 API (no torch needed)."""
import contextlib
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from samotate.core.sam import Sam3Backend, SamNotAvailable, mask_to_xyxy, pick_device


class FakeProcessor:
    """Mimics sam3.model.sam3_image_processor.Sam3Processor."""

    def __init__(self):
        self.encodes = 0
        self.confidence_threshold = 0.5
        self.boxes = np.array([[1, 2, 30, 40], [5, 5, 9, 9]], dtype=np.float32)
        self.scores = np.array([0.9, 0.3], dtype=np.float32)

    def set_image(self, image, state=None):
        self.encodes += 1
        return {"backbone_out": {"img": image.size}}

    def reset_all_prompts(self, state):
        state.pop("boxes", None)
        state.pop("scores", None)

    def set_confidence_threshold(self, threshold, state=None):
        self.confidence_threshold = threshold

    def set_text_prompt(self, prompt, state):
        keep = self.scores > self.confidence_threshold  # filtering happens in here
        state["boxes"], state["scores"] = self.boxes[keep], self.scores[keep]
        return state


class FakeModel:
    def predict_inst(self, state, point_coords, point_labels, multimask_output):
        masks = np.zeros((3, 20, 20), dtype=bool)
        masks[0, 2:5, 3:8] = True     # low score
        masks[1, 0:10, 0:10] = True   # best
        # masks[2] stays empty and must be dropped
        return masks, np.array([0.2, 0.8, 0.5]), None


def _backend(points=True):
    b = Sam3Backend.__new__(Sam3Backend)  # skip the torch/sam3 imports
    b.device, b._points = "cpu", points
    b._processor, b._model = FakeProcessor(), FakeModel()
    b._torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
    b._lock, b._image, b._state = threading.Lock(), None, None
    return b


def test_image_encoded_once_across_prompts():
    b = _backend()
    img = np.zeros((20, 20, 3), "uint8")
    for prompt in ("a", "b", "c"):
        b.text_prompt(img, prompt, 0.5)
    assert b._processor.encodes == 1
    b.text_prompt(np.zeros((20, 20, 3), "uint8"), "a", 0.5)  # new image → re-encode
    assert b._processor.encodes == 2


def test_prompt_threshold_applied_inside_processor():
    b = _backend()
    img = np.zeros((20, 20, 3), "uint8")
    assert len(b.text_prompt(img, "x", 0.5)) == 1
    # below the old fixed 0.4 load-time threshold: must now reach the processor
    hits = b.text_prompt(img, "x", 0.2)
    assert b._processor.confidence_threshold == 0.2
    assert [round(h["score"], 2) for h in hits] == [0.9, 0.3]


def test_point_prompt_returns_mask_boxes_best_first():
    hits = _backend().point_prompt(np.zeros((20, 20, 3), "uint8"), 5, 5)
    assert [h["bbox_xyxy"] for h in hits] == [[0, 0, 10, 10], [3, 2, 8, 5]]
    assert hits[0]["score"] == pytest.approx(0.8)


def test_point_prompt_needs_point_head():
    with pytest.raises(SamNotAvailable):
        _backend(points=False).point_prompt(np.zeros((4, 4, 3), "uint8"), 1, 1)


def test_mask_to_xyxy_empty():
    assert mask_to_xyxy(np.zeros((4, 4), bool)) is None


def _torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


@pytest.mark.parametrize("cuda,mps,expected", [
    (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu"),
])
def test_pick_device_prefers_cuda_then_mps(cuda, mps, expected, monkeypatch):
    monkeypatch.delenv("SAMOTATE_DEVICE", raising=False)
    assert pick_device(_torch(cuda, mps)) == expected


def test_pick_device_env_override(monkeypatch):
    monkeypatch.setenv("SAMOTATE_DEVICE", "cpu")
    assert pick_device(_torch(cuda=True)) == "cpu"
    monkeypatch.setenv("SAMOTATE_DEVICE", "cuda")
    with pytest.raises(RuntimeError, match="not available"):
        pick_device(_torch(cuda=False))
