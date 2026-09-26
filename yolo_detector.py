"""YOLO11 detector and thread-safe background worker."""
from dataclasses import dataclass
import shutil
import threading
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

import config


@dataclass
class YoloSnapshot:
    person_boxes: list
    phone_boxes: list
    timestamp: float
    error: str | None = None


class YoloWorker:
    def __init__(self, model, scale, interval_sec, conf_fn):
        self.model = model
        self.scale = scale
        self.interval_sec = interval_sec
        self.conf_fn = conf_fn
        self._lock = threading.Lock()
        self._frame = None
        self._night = False
        self._snapshot = YoloSnapshot([], [], 0.0, None)
        self._error_count = 0
        self._last_error_ts = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="YOLOWorker", daemon=True)
        self._thread.start()

    def set_frame(self, frame, night_active):
        if frame is None:
            return
        with self._lock:
            self._frame = frame.copy()
            self._night = bool(night_active)

    def clear_results(self):
        with self._lock:
            self._frame = None
            self._snapshot = YoloSnapshot([], [], 0.0, "cleared")

    def snapshot(self):
        with self._lock:
            return YoloSnapshot(
                list(self._snapshot.person_boxes),
                list(self._snapshot.phone_boxes),
                self._snapshot.timestamp,
                self._snapshot.error,
            )

    def is_stale(self):
        snap = self.snapshot()
        return snap.timestamp <= 0 or (time.monotonic() - snap.timestamp) > config.YOLO_STALE_AFTER_SEC

    def health(self):
        snap = self.snapshot()
        age = float("inf") if snap.timestamp <= 0 else max(0.0, time.monotonic() - snap.timestamp)
        return {
            "ok": snap.error is None and age <= config.YOLO_STALE_AFTER_SEC,
            "age": age,
            "errors": self._error_count,
            "error": snap.error,
        }

    def _run(self):
        while not self._stop.is_set():
            frame = None
            night = False
            with self._lock:
                if self._frame is not None:
                    frame = self._frame
                    night = self._night
                    self._frame = None
            if frame is not None:
                try:
                    person_conf, phone_conf = self.conf_fn(night)
                    persons, phones = run_yolo_scaled(
                        self.model, frame, self.scale, person_conf, phone_conf,
                        imgsz=config.YOLO_IMAGE_SIZE, device=config.YOLO_DEVICE,
                    )
                    with self._lock:
                        self._snapshot = YoloSnapshot(persons, phones, time.monotonic(), None)
                except Exception as exc:
                    self._error_count += 1
                    now = time.monotonic()
                    with self._lock:
                        self._snapshot = YoloSnapshot([], [], 0.0, str(exc))
                    if now - self._last_error_ts >= config.YOLO_ERROR_LOG_INTERVAL_SEC:
                        self._last_error_ts = now
                        print(f"[WARN] YOLO worker degraded: {exc}")
            self._stop.wait(self.interval_sec)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def load_yolo_model():
    """Load YOLO11n. Ultralytics downloads the pretrained asset on first use."""
    path = Path(config.YOLO_MODEL_PATH)
    if path.exists():
        print(f"[INFO] Loading YOLO model: {path}")
        return YOLO(str(path))
    print(f"[INFO] Loading {config.YOLO_MODEL_NAME}; first run may download the weights.")
    model = YOLO(config.YOLO_MODEL_NAME)
    # Best-effort local copy for offline future runs.
    try:
        source = Path(getattr(model, "ckpt_path", ""))
        if source.exists() and source.resolve() != path.resolve():
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, path)
    except Exception:
        pass
    return model


def run_yolo_scaled(model, frame, scale, conf_threshold, phone_conf_threshold, imgsz=640, device=None):
    if frame is None or frame.size == 0:
        return [], []
    if scale < 0.999:
        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        inv = 1.0 / scale
    else:
        small = frame
        inv = 1.0

    results = model.predict(
        source=small,
        imgsz=imgsz,
        conf=min(conf_threshold, phone_conf_threshold),
        device=device,
        verbose=False,
        max_det=20,
    )
    result = results[0]
    person_boxes, phone_boxes = [], []
    if result.boxes is None:
        return person_boxes, phone_boxes

    for item in result.boxes:
        conf = float(item.conf.item())
        label = int(item.cls.item())
        required = phone_conf_threshold if label == config.COCO_CELL_PHONE_CLASS else conf_threshold
        if conf < required:
            continue
        coords = [float(x) * inv for x in item.xyxy[0].tolist()]
        if label == config.COCO_PERSON_CLASS:
            person_boxes.append(coords)
        elif label == config.COCO_CELL_PHONE_CLASS:
            phone_boxes.append(coords)
    return person_boxes, phone_boxes


def confidence_thresholds_for_mode(night_active):
    if night_active:
        return config.YOLO_PERSON_CONF_NIGHT, config.YOLO_PHONE_CONF_NIGHT
    return config.YOLO_PERSON_CONF, config.YOLO_PHONE_CONF
