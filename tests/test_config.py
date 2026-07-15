import pytest
from pydantic import ValidationError

from samotate.config import ClassDef, ProjectConfig, Prompt


def _classes():
    return [ClassDef(id=0, name="a"), ClassDef(id=1, name="b")]


def test_requires_a_source():
    with pytest.raises(ValidationError):
        ProjectConfig(classes=_classes())


def test_rejects_both_sources(tmp_path):
    with pytest.raises(ValidationError):
        ProjectConfig(video=tmp_path / "v.mp4", images_dir=tmp_path, classes=_classes())


def test_duplicate_class_ids_rejected(tmp_path):
    with pytest.raises(ValidationError):
        ProjectConfig(
            images_dir=tmp_path,
            classes=[ClassDef(id=0, name="a"), ClassDef(id=0, name="b")],
        )


def test_hierarchical_requires_base_class(tmp_path):
    with pytest.raises(ValidationError):
        ProjectConfig(
            images_dir=tmp_path,
            classes=_classes(),
            reconcile={"strategy": "hierarchical"},
        )


def test_threshold_bounds():
    with pytest.raises(ValidationError):
        Prompt(text="x", threshold=1.5)


def test_yaml_roundtrip(tmp_path):
    cfg = ProjectConfig(
        images_dir=tmp_path / "imgs",
        root=tmp_path / "out",
        classes=[ClassDef(id=0, name="cat", prompts=[Prompt(text="a cat")])],
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    loaded = ProjectConfig.from_yaml(path)
    assert loaded.classes[0].name == "cat"
    assert loaded.classes[0].prompts[0].text == "a cat"
    assert loaded.frames_dir == loaded.root / "frames"
