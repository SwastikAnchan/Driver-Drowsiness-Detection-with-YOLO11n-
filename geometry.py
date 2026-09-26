"""Pure geometry/math helpers. No I/O."""
import math
import numpy as np
from config import MOUTH_BOTTOM_IDX, MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX, MOUTH_TOP_IDX


def _dist(p1, p2):
    return float(math.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def landmarks_to_px(face_landmarks, w, h):
    return [(float(lm.x) * w, float(lm.y) * h) for lm in face_landmarks]


def eye_aspect_ratio(pts_px, eye_idx):
    if len(pts_px) <= max(eye_idx):
        return 0.0
    p1, p2, p3, p4, p5, p6 = [pts_px[i] for i in eye_idx]
    horizontal = _dist(p1, p4)
    if horizontal <= 1e-9:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(pts_px):
    if len(pts_px) <= max(MOUTH_TOP_IDX + MOUTH_BOTTOM_IDX + [MOUTH_LEFT_IDX, MOUTH_RIGHT_IDX]):
        return 0.0
    horizontal = _dist(pts_px[MOUTH_LEFT_IDX], pts_px[MOUTH_RIGHT_IDX])
    if horizontal <= 1e-9:
        return 0.0
    verticals = [_dist(pts_px[t], pts_px[b]) for t, b in zip(MOUTH_TOP_IDX, MOUTH_BOTTOM_IDX)]
    return sum(verticals) / (3.0 * horizontal)


def face_bbox_from_landmarks(pts_px, frame_w, frame_h, pad_ratio=0.08):
    if not pts_px:
        return 0, 0, 0, 0
    xs = [p[0] for p in pts_px]
    ys = [p[1] for p in pts_px]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad_x = (x_max - x_min) * pad_ratio
    pad_y = (y_max - y_min) * pad_ratio
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(frame_w, int(x_max + pad_x))
    y2 = min(frame_h, int(y_max + pad_y))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def box_area(box):
    if box is None or len(box) != 4:
        return 0.0
    return max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))


def box_iou(box1, box2):
    if box1 is None or box2 is None:
        return 0.0
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = box_area(box1) + box_area(box2) - inter
    return inter / union if union > 0 else 0.0


def box_containment_ratio(inner_box, outer_box):
    if inner_box is None or outer_box is None:
        return 0.0
    x1 = max(float(inner_box[0]), float(outer_box[0]))
    y1 = max(float(inner_box[1]), float(outer_box[1]))
    x2 = min(float(inner_box[2]), float(outer_box[2]))
    y2 = min(float(inner_box[3]), float(outer_box[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    inner_area = box_area(inner_box)
    return inter / inner_area if inner_area > 0 else 0.0


def box_center(box):
    if box is None:
        return (0.0, 0.0)
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def expand_box(box, ratio, frame_w=None, frame_h=None):
    if box is None:
        return None
    w = max(0.0, float(box[2] - box[0]))
    h = max(0.0, float(box[3] - box[1]))
    x1 = float(box[0]) - w * ratio
    y1 = float(box[1]) - h * ratio
    x2 = float(box[2]) + w * ratio
    y2 = float(box[3]) + h * ratio
    if frame_w is not None:
        x1, x2 = max(0.0, x1), min(float(frame_w), x2)
    if frame_h is not None:
        y1, y2 = max(0.0, y1), min(float(frame_h), y2)
    return [x1, y1, x2, y2]


def is_driver_box(box, w_max, driver_side):
    cx, _ = box_center(box)
    if driver_side == "right":
        return cx >= w_max * 0.55
    return cx <= w_max * 0.45


def driver_side_score(box, w_max, driver_side):
    """0..1 score for how strongly a box lies on the configured driver side."""
    cx, _ = box_center(box)
    ratio = cx / max(1.0, float(w_max))
    if driver_side == "right":
        return min(1.0, max(0.0, (ratio - 0.45) / 0.55))
    return min(1.0, max(0.0, (0.55 - ratio) / 0.55))
