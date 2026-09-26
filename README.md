# Driver Monitoring System — Hardened Build

Camera display, fullscreen dashboard, driver-side detection, face landmarks,
drowsiness/eye closure, yawning, head-turn distraction, phone detection, hand
visualization, night enhancement, and CSV event logging. Voice is the only
alert channel — there is no beep path.

One OpenCV display window. Press **Q** or **Esc** to exit.

## Accuracy note

No software-only rewrite can guarantee a 90–95% real-world accuracy rate
without measuring the system on labeled data from the actual camera,
mounting position, lighting, users, and failure cases. The included quality
gate and training tools exist so that target is measured, not assumed.

The default detector is **YOLO11n** (Ultralytics 8.3.0), with a published
COCO mAP50-95 of 39.5 at 640 px. That benchmark is not the same as a
90–95% driver-phone event accuracy number — see "Reaching 90–95%" below.

## Folder layout

```text
driver_monitoring_final/
  main.py
  app.py
  config.py
  yolo_detector.py
  face_analyzer.py
  hand_analyzer.py
  geometry.py
  temporal_state.py
  night_vision.py
  camera_utils.py
  audio_alert.py
  event_logger.py
  hud.py
  health_check.py
  train_yolo.py
  evaluate_yolo.py
  requirements.txt
  event_history_log.csv   (created at runtime)
  models/
    face_landmarker.task
    hand_landmarker.task
    yolo11n.pt             (downloaded on first run if not already present)
  dataset/
    data.yaml
    README.txt
    images/train/
    images/val/
    labels/train/
    labels/val/
  tests/
    test_geometry.py
    test_temporal.py
```

## Manual setup (Windows)

Use Python 3.12 for the most predictable install on this stack.
```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python health_check.py
python main.py
```

The first run may download the YOLO11n weights. The face and hand MediaPipe
task files are already included in this package.

### Useful flags

```bat
:: Use a specific camera
python main.py --camera-index 1

:: Force a different Windows camera backend
python main.py --camera-index 0 --backend msmf

:: Normal resizable window instead of fullscreen
python main.py --windowed
```

## What was fixed

- **Timers are time-based, not frame-based** — changing camera FPS no longer
  silently changes the meaning of the drowsiness/yawn/distraction thresholds.
- **YOLO runs in a background worker** with a bounded "latest frame only"
  design. Old results expire, camera reconnection clears stale detections,
  and inference errors are recorded as a degraded state instead of a false
  "nothing detected" state.
- **Phone detection** uses phone-box containment inside the driver region
  rather than plain IoU, requires repeated fresh YOLO confirmations, and
  clears only after repeated misses — reducing one-frame false alarms.
- **Voice alerts** use one dedicated worker and a bounded queue: no
  one-thread-per-warning, and no beep implementation anywhere in the project.
- **Face model download** uses a timeout and atomic temp-file replacement,
  so an interrupted download can't leave a corrupt model file that looks
  valid.
- **CSV logging is non-fatal** — disk/full-file-lock errors are reported
  without taking down the camera loop.
- **`app.py`** is now only a compatibility wrapper around `main.py`, so
  there is one authoritative implementation.
- **Hand display** is retained, but no longer claims detected hands mean
  "hands off wheel" (that branch was logically reversed). The build shows
  driver/passenger hands and does not claim steering-wheel contact, since
  it doesn't detect the wheel itself.

## Reaching 90–95%

Fine-tune the detector on your own cabin-camera data, then validate on a
held-out test set plus a separate real-driving test set. `dataset/README.txt`
explains the expected YOLO dataset format.

```bat
python train_yolo.py --data dataset/data.yaml --model yolo11n.pt --device 0 --epochs 100
python evaluate_yolo.py --data dataset/data.yaml --model runs/detect/train/weights/best.pt --device 0 --min-map50 0.90
```

The `--min-map50 0.90` gate makes evaluation fail when mAP50 is below 90%.
It doesn't make the live system 90% by itself — it prevents the project from
being called "90% accurate" without a measured validation result.

## Licensing

This project uses the Ultralytics package and pretrained YOLO11 weights.
Ultralytics documents YOLO11 under AGPL-3.0 and Enterprise licensing —
review the applicable terms for your intended use or commercial deployment.