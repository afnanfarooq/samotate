from samotate.models import Annotation, Box


def test_yolo_clips_boxes_to_image():
    # overshoots every edge; Ultralytics drops labels outside [0, 1]
    line = Box(class_id=0, xyxy=(-5, -5, 105, 55)).to_yolo(100, 50)
    _, cx, cy, w, h = line.split()
    assert (float(cx), float(cy), float(w), float(h)) == (0.5, 0.5, 1.0, 1.0)


def test_yolo_normalizes_flipped_corners():
    assert Box(class_id=0, xyxy=(30, 40, 10, 20)).to_yolo(100, 100) == (
        Box(class_id=0, xyxy=(10, 20, 30, 40)).to_yolo(100, 100)
    )


def test_yolo_skips_boxes_with_no_area():
    ann = Annotation(
        image="a.png", width=100, height=100,
        boxes=[
            Box(class_id=0, xyxy=(10, 10, 10, 50)),     # zero width
            Box(class_id=0, xyxy=(150, 10, 200, 50)),   # fully outside
            Box(class_id=0, xyxy=(10, 10, 20, 20)),
        ],
    )
    assert len(ann.to_yolo_lines()) == 1
