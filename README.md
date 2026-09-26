# Driver Monitoring System — Hardened Build

This version keeps the original project behavior—camera display, fullscreen dashboard, driver-side detection, face landmarks, drowsiness/eye closure, yawning, head-turn distraction, phone detection, hand visualization, night enhancement, and CSV event logging—but removes the beep path and makes voice the only alert channel.

There is exactly one OpenCV display window. Press **Q** or **Esc** to exit.

## Important accuracy point

No software-only rewrite can honestly guarantee a 90–95% real-world accuracy rate without measuring the system on labeled data from the actual camera, mounting position, lighting, users, and failure cases. The included quality gate and training tools are there so the target is measured rather than assumed.

The default detector is **YOLO11n**, introduced in the Ultralytics 8.3.0 release and documented by Ultralytics with a COCO mAP50-95 of 39.5 at 640 px. That published benchmark is not the same thing as a 90–95% driver-phone event accuracy number. citeturn925365search3turn925365search0

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
  setup_windows.bat
  run.bat
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

## Windows setup

Use Python 3.12 for the most predictable installation path for this stack.

1. Extract this folder.
2. Open Command Prompt in the project folder.
3. Run:

```bat
setup_windows.bat
```

4. Then run:

```bat
run.bat
```

The first run may download the YOLO11n weights. The face and hand MediaPipe task files are already included in this package.

## Manual setup

```bat
py -3.12 -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python health_check.py
python main.py
```

Use a specific camera if needed:

```bat
python main.py --camera-index 1
```

Force a different Windows camera backend:

```bat
python main.py --camera-index 0 --backend msmf
```

For a normal resizable window instead of fullscreen:

```bat
python main.py --windowed
```

## What was fixed

The alert timers are based on elapsed time rather than frame count, so changing actual camera FPS does not silently change the meaning of the drowsiness/yawn/distraction thresholds.

YOLO runs in a background worker with a bounded “latest frame only” design. Old results expire. Camera reconnection clears stale detections. YOLO inference errors are recorded as a degraded state instead of being converted to a believable “nothing detected” state.

Phone detection now uses phone-box containment inside the driver region rather than plain IoU. It also requires repeated fresh YOLO confirmations and clears only after repeated misses, reducing one-frame false alarms.

Voice alerts use one dedicated worker and a bounded queue. They do not create one thread per warning and there is no beep implementation anywhere in the final project.

The face model is downloaded with a timeout and atomic temporary-file replacement, so an interrupted download does not leave a corrupt model file that looks valid.

CSV logging is non-fatal. Disk/full-file-lock errors are reported without taking down the camera loop.

The old `app.py` is now only a compatibility wrapper around `main.py`, so there is one authoritative executable implementation.

The hand display is retained. The previous code claimed that detected hands meant “hands off wheel” under one branch, which was logically reversed. The new build displays driver/passenger hands and, by default, does not make a safety claim about steering-wheel contact because it does not actually detect the steering wheel.

## 90–95% project target

The reliable route to a measured 90–95% result is to fine-tune the detector on your own cabin-camera data and then validate it on a held-out test set plus a separate real-driving test set.

The included `dataset/README.txt` explains the expected YOLO dataset. After collecting labels, train for example:

```bat
python train_yolo.py --data dataset/data.yaml --model yolo11n.pt --device 0 --epochs 100
```

Then evaluate:

```bat
python evaluate_yolo.py --data dataset/data.yaml --model runs/detect/train/weights/best.pt --device 0 --min-map50 0.90
```

A quality gate of 0.90 means the evaluation command fails when mAP50 is below 90%. That does not magically make the live system 90%; it prevents the project from being called “90% accurate” without a measured validation result.

## Ultralytics licensing

The project uses the Ultralytics package and pretrained YOLO11 weights. Ultralytics documents YOLO11 under AGPL-3.0 and Enterprise licensing; review the applicable terms for your intended project or commercial deployment. citeturn572253search0turn572253search3
