from samotate.core.geometry import iou_xyxy, nms


def test_iou_identical():
    assert iou_xyxy([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_iou_disjoint():
    assert iou_xyxy([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_iou_half_overlap():
    # two 10x10 boxes overlapping in a 5x10 region: inter=50, union=150
    assert iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == 50 / 150


def test_nms_suppresses_overlap():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
    scores = [0.9, 0.8, 0.7]
    keep = nms(boxes, scores, iou_threshold=0.5)
    # the second box overlaps the first heavily and is dropped
    assert keep == [0, 2]
