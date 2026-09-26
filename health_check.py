"""Installation and file health check. Safe to run before main.py."""
from importlib import import_module
from pathlib import Path
import platform
import sys

import config

REQUIRED = ["cv2", "numpy", "mediapipe", "ultralytics", "pyttsx3"]


def main():
    print("=== Driver Monitoring Health Check ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    ok = True
    for mod in REQUIRED:
        try:
            imported = import_module(mod)
            version = getattr(imported, "__version__", "unknown")
            print(f"[OK] {mod}: {version}")
        except Exception as exc:
            ok = False
            print(f"[FAIL] {mod}: {exc}")

    for label, path in (
        ("face model", config.FACE_MODEL_PATH),
        ("hand model", config.HAND_MODEL_PATH),
    ):
        p = Path(path)
        if p.exists() and p.stat().st_size > 100_000:
            print(f"[OK] {label}: {p}")
        else:
            ok = False
            print(f"[FAIL] {label}: {p}")

    yolo = Path(config.YOLO_MODEL_PATH)
    if yolo.exists():
        print(f"[OK] YOLO weights: {yolo}")
    else:
        print(f"[INFO] YOLO weights not present yet. First run will load {config.YOLO_MODEL_NAME}.")

    print("[INFO] Voice mode: voice only; no beep implementation is enabled.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
