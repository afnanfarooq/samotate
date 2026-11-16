"""End-to-end engine test using a mock SAM backend (no model required)."""
from samotate.config import ClassDef, ProjectConfig, ReconcileConfig
from samotate.core import detect, export, extract, reconcile
from samotate.models import Annotation


def test_index_images(image_project):
    frames = extract.gather_source_images(image_project)
    assert len(frames) == 3
    assert all(p.exists() for p in frames)


def test_detect_and_reconcile_per_prompt(image_project, mock_backend):
    extract.gather_source_images(image_project)
    n = detect.detect_all(image_project, backend=mock_backend)
    assert n == 3
    # one detection json per frame
    assert len(list(image_project.detections_dir.glob("*.json"))) == 3

    reconcile.reconcile_all(image_project)
    ann_files = list(image_project.annotations_dir.glob("*.json"))
    assert len(ann_files) == 3
    ann = Annotation.load(ann_files[0])
    assert len(ann.boxes) == 1
    assert ann.boxes[0].class_id == 0
    assert ann.boxes[0].source == "auto"


def test_export_writes_labels_and_yaml(image_project, mock_backend):
    extract.gather_source_images(image_project)
    detect.detect_all(image_project, backend=mock_backend)
    reconcile.reconcile_all(image_project)
    result = export.export_dataset(image_project)

    labels = list(image_project.labels_dir.glob("*.txt"))
    assert len(labels) == 3
    # YOLO line: class cx cy w h, all normalized into [0,1]
    line = labels[0].read_text().strip().split("\n")[0].split()
    assert line[0] == "0"
    assert all(0.0 <= float(v) <= 1.0 for v in line[1:])

    data_yaml = (image_project.root / "data.yaml").read_text()
    assert "object" in data_yaml
    assert result["train"] + result["val"] == 3


def test_reconcile_preserves_reviewed_annotations(image_project, mock_backend):
    extract.gather_source_images(image_project)
    detect.detect_all(image_project, backend=mock_backend)
    reconcile.reconcile_all(image_project)

    # simulate a human review that empties an annotation
    ann_files = sorted(image_project.annotations_dir.glob("*.json"))
    ann = Annotation.load(ann_files[0])
    ann.boxes = []
    ann.reviewed = True
    ann.save(ann_files[0])

    # re-running reconcile must not clobber the reviewed file
    reconcile.reconcile_all(image_project)
    assert Annotation.load(ann_files[0]).boxes == []


def test_hierarchical_strategy(tmp_path):
    """Base 'customer' subdivided by 'staff', overridden by 'seated'."""
    import numpy as np
    from PIL import Image

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    Image.fromarray(np.zeros((100, 100, 3), "uint8")).save(images_dir / "a.png")

    cfg = ProjectConfig(
        images_dir=images_dir,
        root=tmp_path / "out",
        classes=[
            ClassDef(id=0, name="staff"),
            ClassDef(id=1, name="customer"),
            ClassDef(id=2, name="seated"),
        ],
        reconcile=ReconcileConfig(
            strategy="hierarchical",
            base_class="customer",
            override_classes=["seated"],
            iou_threshold=0.4,
            override_iou=0.5,
        ),
    )
    # two persons; one overlaps a staff box → staff, the other is a plain customer
    meta = {
        "frame": "a.png",
        "width": 100,
        "height": 100,
        "detections": {
            "customer": [
                {"bbox_xyxy": [0, 0, 20, 40], "score": 0.9},
                {"bbox_xyxy": [60, 60, 80, 95], "score": 0.9},
            ],
            "staff": [{"bbox_xyxy": [0, 0, 20, 40], "score": 0.8}],
            "seated": [],
        },
    }
    ann = reconcile.reconcile_meta(cfg, meta)
    class_ids = sorted(b.class_id for b in ann.boxes)
    assert class_ids == [0, 1]  # one staff, one customer


# --- regression tests -------------------------------------------------------

import pytest  # noqa: E402
import yaml  # noqa: E402

from samotate.core.sam import NullBackend, SamNotAvailable  # noqa: E402
from samotate.errors import SamotateError  # noqa: E402


def _write_video(path, n_frames=10, fps=10.0):
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    for i in range(n_frames):
        writer.write(np.full((48, 64, 3), i * 20, "uint8"))
    writer.release()


def test_video_extraction_samples_by_stride(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video, n_frames=10, fps=10.0)
    cfg = ProjectConfig(video=video, root=tmp_path / "out", fps=5.0,
                        classes=[ClassDef(id=0, name="a")])
    frames = extract.gather_source_images(cfg)
    assert [p.name for p in frames] == [f"frame_{i:07d}.jpg" for i in (0, 2, 4, 6, 8)]


def test_video_extraction_respects_limit(tmp_path):
    video = tmp_path / "clip.avi"
    _write_video(video)
    cfg = ProjectConfig(video=video, root=tmp_path / "out", fps=10.0, limit=3,
                        classes=[ClassDef(id=0, name="a")])
    assert len(extract.gather_source_images(cfg)) == 3


def test_missing_video_is_a_clear_error(tmp_path):
    cfg = ProjectConfig(video=tmp_path / "nope.mp4", root=tmp_path / "out",
                        classes=[ClassDef(id=0, name="a")])
    with pytest.raises(SamotateError, match="video not found"):
        extract.gather_source_images(cfg)


def test_missing_images_dir_is_a_clear_error(tmp_path):
    cfg = ProjectConfig(images_dir=tmp_path / "nope", root=tmp_path / "out",
                        classes=[ClassDef(id=0, name="a")])
    with pytest.raises(SamotateError, match="images_dir not found"):
        extract.gather_source_images(cfg)


def test_duplicate_stems_rejected(image_project):
    src = image_project.images_dir / "img_000.png"
    (image_project.images_dir / "img_000.jpg").write_bytes(src.read_bytes())
    with pytest.raises(SamotateError, match="share a file stem"):
        extract.gather_source_images(image_project)


def test_detect_without_sam_fails_loudly(image_project):
    extract.gather_source_images(image_project)
    with pytest.raises(SamNotAvailable):
        detect.detect_all(image_project, backend=NullBackend())
    assert not list(image_project.detections_dir.glob("*.json"))


def test_detect_without_prompts_fails(image_project, mock_backend):
    extract.gather_source_images(image_project)
    cfg = image_project.model_copy(update={"classes": [ClassDef(id=0, name="object")]})
    with pytest.raises(SamotateError, match="no class has any prompts"):
        detect.detect_all(cfg, backend=mock_backend)


def test_detect_skips_backend_load_when_all_cached(image_project, mock_backend, monkeypatch):
    extract.gather_source_images(image_project)
    detect.detect_all(image_project, backend=mock_backend)

    def boom(**_):
        raise AssertionError("model should not load when nothing to do")

    monkeypatch.setattr(detect, "load_backend", boom)
    assert detect.detect_all(image_project) == 0


def test_hierarchical_dedups_multi_prompt_hits(tmp_path):
    """Two staff prompts firing on one person must yield one staff box."""
    cfg = ProjectConfig(
        images_dir=tmp_path,
        classes=[ClassDef(id=0, name="staff"), ClassDef(id=1, name="customer")],
        reconcile=ReconcileConfig(strategy="hierarchical", base_class="customer"),
    )
    meta = {
        "frame": "a.png", "width": 100, "height": 100,
        "detections": {
            "customer": [{"bbox_xyxy": [0, 0, 20, 40], "score": 0.9},
                         {"bbox_xyxy": [1, 0, 21, 40], "score": 0.8}],
            "staff": [{"bbox_xyxy": [0, 0, 20, 40], "score": 0.8},
                      {"bbox_xyxy": [0, 1, 20, 41], "score": 0.7}],
        },
    }
    ann = reconcile.reconcile_meta(cfg, meta)
    assert [b.class_id for b in ann.boxes] == [0]


def test_reconcile_without_detections_is_a_clear_error(image_project):
    with pytest.raises(SamotateError, match="run the 'detect' stage"):
        reconcile.reconcile_all(image_project)


def test_split_uses_only_labeled_frames(tmp_path):
    """Partially annotated projects must still get a val split."""
    import numpy as np
    from PIL import Image

    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    for i in range(10):
        Image.fromarray(np.zeros((10, 10, 3), "uint8")).save(images_dir / f"f{i:02d}.png")
    cfg = ProjectConfig(images_dir=images_dir, root=tmp_path / "out",
                        classes=[ClassDef(id=0, name="a")], train_fraction=0.5)
    extract.gather_source_images(cfg)
    cfg.annotations_dir.mkdir(parents=True)
    for i in range(4):  # only the first 4 of 10 frames annotated
        Annotation(image=f"f{i:02d}.png", width=10, height=10).save(
            cfg.annotations_dir / f"f{i:02d}.json"
        )
    result = export.export_dataset(cfg)
    assert (result["train"], result["val"]) == (2, 2)


def test_data_yaml_survives_special_class_names(image_project, mock_backend):
    cfg = image_project.model_copy(update={"classes": [
        ClassDef(id=0, name="hard: hat #1", prompts=[{"text": "x"}]),
    ]})
    extract.gather_source_images(cfg)
    detect.detect_all(cfg, backend=mock_backend)
    reconcile.reconcile_all(cfg)
    export.export_dataset(cfg)
    data = yaml.safe_load((cfg.root / "data.yaml").read_text())
    assert data["names"] == {0: "hard: hat #1"}
    assert data["train"] == "images/train"
