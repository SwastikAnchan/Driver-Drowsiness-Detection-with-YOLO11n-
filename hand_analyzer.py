"""MediaPipe hand detection and driver/passenger hand classification."""
import os

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config
from geometry import box_containment_ratio, box_iou, landmarks_to_px

# Canonical MediaPipe hand graph. Avoids depending on mp.solutions.hands in newer releases.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class HandsFrameResult:
    __slots__ = ("hands_detected", "landmarks_list", "error")

    def __init__(self):
        self.hands_detected = 0
        self.landmarks_list = []
        self.error = None


def ensure_model_exists():
    if not os.path.exists(config.HAND_MODEL_PATH):
        raise FileNotFoundError(
            f"Missing hand model: {config.HAND_MODEL_PATH}. "
            "Copy hand_landmarker.task into the models folder."
        )


def create_hand_landmarker():
    ensure_model_exists()
    base_options = mp_python.BaseOptions(
        model_asset_path=str(config.HAND_MODEL_PATH),
        delegate=mp_python.BaseOptions.Delegate.CPU,
    )
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=config.HAND_MAX_COUNT,
        min_hand_detection_confidence=config.HAND_MIN_DETECTION_CONF,
        min_tracking_confidence=config.HAND_MIN_TRACKING_CONF,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def analyze_hands(hand_landmarker, mp_image, timestamp_ms, w, h):
    result = HandsFrameResult()
    try:
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
    except Exception as exc:
        result.error = str(exc)
        return result
    for hand_landmarks in (hand_result.hand_landmarks or []):
        try:
            result.landmarks_list.append(landmarks_to_px(hand_landmarks, w, h))
        except Exception:
            continue
    result.hands_detected = len(result.landmarks_list)
    return result


def hand_bbox_from_landmarks(points):
    if not points:
        return [0, 0, 0, 0]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def classify_driver_hands(hand_result, driver_box):
    if hand_result is None:
        return [], []
    if driver_box is None:
        return [], [(pts, hand_bbox_from_landmarks(pts)) for pts in hand_result.landmarks_list]

    driver_hands, passenger_hands = [], []
    for pts in hand_result.landmarks_list:
        bbox = hand_bbox_from_landmarks(pts)
        iou = box_iou(bbox, driver_box)
        containment = box_containment_ratio(bbox, driver_box)
        if iou > 0.05 or containment > 0.15:
            driver_hands.append((pts, bbox))
        else:
            passenger_hands.append((pts, bbox))
    return driver_hands, passenger_hands


def draw_hands(frame, hands, color=(0, 255, 0)):
    for points in hands:
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                x1, y1 = points[a]
                x2, y2 = points[b]
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        x1, y1, x2, y2 = hand_bbox_from_landmarks(points)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
