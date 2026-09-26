"""Robust OpenCV camera discovery and reconnect helpers."""
import platform
import time
import cv2

import config
from night_vision import apply_day_camera_settings, apply_night_camera_settings


def _backend_options(preferred_backend=None):
    system = platform.system()
    if system != "Windows":
        return [("default", None)]
    all_backends = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("default", None)]
    if preferred_backend:
        preferred_backend = preferred_backend.lower()
        selected = [(tag, backend) for tag, backend in all_backends if tag == preferred_backend]
        selected += [(tag, backend) for tag, backend in all_backends if tag != preferred_backend]
        return selected
    return all_backends


def _valid_frame(ret, frame):
    return bool(ret and frame is not None and getattr(frame, "size", 0) > 0)


def _open_once(index, backend):
    cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        try:
            cap.release()
        except Exception:
            pass
        return None
    time.sleep(config.CAMERA_OPEN_DELAY_SEC)

    # Validate native delivery first. Some webcams report opened=True but produce no frames.
    native_ok = 0
    for _ in range(6):
        ret, frame = cap.read()
        if _valid_frame(ret, frame):
            native_ok += 1
        time.sleep(0.04)
    if native_ok == 0:
        cap.release()
        return None

    # Request target properties only after native validation.
    for prop, value in (
        (cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH),
        (cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT),
        (cv2.CAP_PROP_FPS, config.CAM_FPS),
        (cv2.CAP_PROP_BUFFERSIZE, 1),
    ):
        try:
            cap.set(prop, value)
        except Exception:
            pass

    time.sleep(0.20)
    forced_ok = 0
    for _ in range(6):
        ret, frame = cap.read()
        if _valid_frame(ret, frame):
            forced_ok += 1
        time.sleep(0.04)
    if forced_ok == 0:
        # Re-open in native mode instead of keeping a wedged handle.
        cap.release()
        time.sleep(0.15)
        cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            return None
        time.sleep(0.20)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    print(f"[INFO] Camera ready: index={index}, backend={backend}, {width}x{height}, reported FPS={fps:.1f}")
    return cap


def open_camera(preferred_index=None, preferred_backend=None):
    indices = []
    if preferred_index is not None:
        try:
            indices.append(int(preferred_index))
        except (ValueError, TypeError):
            pass
    for idx in range(config.CAMERA_SCAN_MAX_INDEX + 1):
        if idx not in indices:
            indices.append(idx)

    for backend_tag, backend in _backend_options(preferred_backend):
        for index in indices:
            print(f"[INFO] Trying camera index={index}, backend={backend_tag}")
            for attempt in range(config.CAMERA_RECONNECT_RETRIES):
                cap = _open_once(index, backend)
                if cap is not None:
                    return cap
                if attempt + 1 < config.CAMERA_RECONNECT_RETRIES:
                    time.sleep(0.15)
    print("[ERROR] No working camera was found.")
    print("[ERROR] Check Windows camera privacy permissions and close other apps using the webcam.")
    return None


def sync_camera_to_light_mode(cap, night_active, current_mode_is_night):
    if cap is None or night_active == current_mode_is_night:
        return current_mode_is_night
    try:
        if night_active:
            apply_night_camera_settings(cap)
        else:
            apply_day_camera_settings(cap)
    except Exception as exc:
        print(f"[WARN] Camera light-mode settings failed: {exc}")
    return bool(night_active)
