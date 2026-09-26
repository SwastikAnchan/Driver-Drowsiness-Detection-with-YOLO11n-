import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry import box_containment_ratio, box_iou, expand_box, is_driver_box


def test_full_containment():
    assert abs(box_containment_ratio([2, 2, 4, 4], [0, 0, 10, 10]) - 1.0) < 1e-9


def test_iou():
    assert abs(box_iou([0, 0, 10, 10], [0, 0, 10, 10]) - 1.0) < 1e-9


def test_expand():
    b = expand_box([10, 10, 20, 20], 0.1)
    assert b == [9.0, 9.0, 21.0, 21.0]


def test_driver_side():
    assert is_driver_box([60, 0, 90, 50], 100, "right")
    assert is_driver_box([10, 0, 30, 50], 100, "left")
