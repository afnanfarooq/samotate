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


def test_relative_paths_resolve_against_config_dir(tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "config.yaml").write_text(
        "images_dir: ./images\nclasses:\n  - {id: 0, name: a}\n", encoding="utf-8"
    )
    # run from somewhere else entirely
    monkeypatch.chdir(tmp_path)
    cfg = ProjectConfig.from_yaml("proj/config.yaml")
    assert cfg.images_dir == proj.resolve() / "images"
    assert cfg.root == proj.resolve() / "dataset"  # default root too


def test_class_ids_must_be_contiguous(tmp_path):
    with pytest.raises(ValidationError, match="no gaps"):
        ProjectConfig(
            images_dir=tmp_path,
            classes=[ClassDef(id=0, name="a"), ClassDef(id=2, name="b")],
        )


@pytest.mark.parametrize(
    "field", [{"fps": 0}, {"train_fraction": 0}, {"train_fraction": 1.5}, {"limit": 0}]
)
def test_numeric_bounds(tmp_path, field):
    with pytest.raises(ValidationError):
        ProjectConfig(images_dir=tmp_path, classes=_classes(), **field)
