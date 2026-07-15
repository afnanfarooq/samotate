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
