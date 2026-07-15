"""Shared test fixtures: a mock SAM backend and a synthetic image project."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from samotate.config import ClassDef, ProjectConfig, Prompt, ReconcileConfig


class MockBackend:
    """A deterministic stand-in for SAM used in tests.

    Returns a fixed box for any text prompt and a small box around a clicked
    point, so detection/assist paths can be exercised without a model.
    """

    def __init__(self, box=(10.0, 10.0, 50.0, 90.0), score=0.9):
        self.box = list(box)
        self.score = score
        self.calls: list[tuple] = []

    def text_prompt(self, image_rgb, prompt, threshold):
        self.calls.append(("text", prompt, threshold))
        return [{"bbox_xyxy": list(self.box), "score": self.score}]

    def point_prompt(self, image_rgb, x, y):
        self.calls.append(("point", x, y))
        return [{"bbox_xyxy": [x - 5, y - 5, x + 5, y + 5], "score": self.score}]


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


@pytest.fixture
def image_project(tmp_path: Path) -> ProjectConfig:
    """A project backed by three synthetic 100x100 images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    for i in range(3):
        arr = (np.random.default_rng(i).integers(0, 255, (100, 100, 3))).astype("uint8")
        Image.fromarray(arr).save(images_dir / f"img_{i:03d}.png")

    return ProjectConfig(
        name="test-project",
        images_dir=images_dir,
        root=tmp_path / "dataset",
        classes=[
            ClassDef(id=0, name="object", prompts=[Prompt(text="an object", threshold=0.5)]),
        ],
        reconcile=ReconcileConfig(strategy="per_prompt"),
        link_mode="copy",
    )
