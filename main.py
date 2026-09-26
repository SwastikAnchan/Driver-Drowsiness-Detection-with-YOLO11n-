"""Main entry point for the hardened driver monitoring system."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import platform
import time

import cv2
import numpy as np

import config
from audio_alert import (
    distraction_alert,
    drowsiness_alert,
    hands_not_visible_alert,
    phone_alert,
    shutdown_voice,
    sleep_alert,
    sleep_and_phone_alert,
    yawn_alert,
)
from camera_utils import open_camera, sync_camera_to_light_mode
from event_logger import EventLogger
from geometry import box_center, box_containment_ratio, box_iou, driver_side_score, expand_box, is_driver_box
from hand_analyzer import analyze_hands, classify_driver_hands, create_hand_landmarker, draw_hands
from face_analyzer import analyze_frame, create_face_landmarker, new_mp_image
from hud import draw_alert_banner, draw_hud_panel, draw_zone_gridlines, get_screen_size, letterbox_to_screen
from night_vision import NightModeState, apply_night_vision_tint, enhance_for_low_light
from temporal_state import DriverState
from yolo_detector import YoloWorker, confidence_thresholds_for_mode, load_yolo_model


def parse_args():
    parser = argparse.ArgumentParser(description="Driver monitoring system")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", choices=["dshow", "msmf", "default"], default=None)
    parser.add_argument("--windowed", action="store_true", help="Use a normal window instead of fullscreen")
    parser.add_argument("--camera-only", action="store_true", help="Camera display without AI models")
    return parser.parse_args()


def driver_side_in_image():
    return config.DRIVER_SIDE if config.FLIP_FRAME else ("left" if config.DRIVER_SIDE == "right" else "right")


def choose_driver_box(person_boxes, face_bbox, w, driver_side):
    candidates = [b for b in person_boxes if is_driver_box(b, w, driver_side)]
    if not candidates:
        return None
    if face_bbox:
        fx, fy, fw, fh = face_bbox
        face_box = [fx, fy, fx + fw, fy + fh]
        inside = sorted(candidates, key=lambda b: box_containment_ratio(face_box, b), reverse=True)
        if inside and box_containment_ratio(face_box, inside[0]) >= 0.25:
            return inside[0]
    return max(candidates, key=lambda b: (driver_side_score(b, w, driver_side), (b[2] - b[0]) * (b[3] - b[1])))


def phone_candidate(phone_box, driver_box, frame_w, frame_h):
    if driver_box is None or phone_box is None:
        return False
    expanded = expand_box(driver_box, config.DRIVER_BOX_EXPANSION, frame_w, frame_h)
    containment = box_containment_ratio(phone_box, expanded)
    cx, cy = box_center(phone_box)
    ex1, ey1, ex2, ey2 = expanded
    inside_center = ex1 <= cx <= ex2 and ey1 <= cy <= ey2
    return containment >= config.PHONE_CONTAINMENT_THRESHOLD and inside_center


def main():
    args = parse_args()
    if platform.system() == "Windows":
        if args.camera_index is None:
            args.camera_index = 0
        if args.backend is None:
            args.backend = "dshow"

    print("[INFO] Starting hardened Driver Monitoring System")
    print(f"[INFO] Driver side: {config.DRIVER_SIDE}, mirrored display: {config.FLIP_FRAME}")
    print("[INFO] Audio mode: VOICE ONLY (all beeps disabled)")

    cap = open_camera(args.camera_index, args.backend)
    if cap is None:
        return 1

    window_name = config.WINDOW_NAME
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if config.FULLSCREEN and not args.windowed:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    screen_w, screen_h = get_screen_size()

    if args.camera_only:
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    frame = np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(frame, "CAMERA RECONNECTING", (30, config.CAM_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                display = letterbox_to_screen(frame, screen_w, screen_h) if not args.windowed else frame
                cv2.imshow(window_name, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return 0

    # Heavy AI imports/models after camera validation.
    face_landmarker = None
    hand_landmarker = None
    yolo_worker = None
    voice_started = True
    logger = EventLogger(config.LOG_FILE)

    try:
        yolo_model = load_yolo_model()
        face_landmarker = create_face_landmarker()
        hand_landmarker = create_hand_landmarker()
        yolo_worker = YoloWorker(yolo_model, config.YOLO_INFER_SCALE, config.YOLO_INTERVAL_SEC, confidence_thresholds_for_mode)

        driver_side = driver_side_in_image()
        state = DriverState()
        night = NightModeState()
        camera_night_mode = False
        bad_frames = 0
        frame_id = 0
        last_ts_ms = 0
        phone_confirms = 0
        phone_misses = 0
        phone_present = False
        last_yolo_ts = 0.0
        last_alert = "System Nominal"
        last_head_pose = "Center Focus"

        while True:
            ret, raw = cap.read()
            now = time.monotonic()
            if not ret or raw is None or raw.size == 0:
                bad_frames += 1
                if bad_frames >= config.CAMERA_BAD_FRAME_LIMIT:
                    print("[WARN] Camera frame stream lost; reconnecting...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    cap = open_camera(args.camera_index, args.backend)
                    if cap is None:
                        print("[ERROR] Camera reconnection failed.")
                        break
                    yolo_worker.clear_results()
                    state.reset_for_camera_reconnect()
                    phone_confirms = phone_misses = 0
                    phone_present = False
                    last_yolo_ts = 0.0
                    bad_frames = 0
                # Keep the same OpenCV window alive while reconnecting.
                placeholder = np.zeros((config.CAM_HEIGHT, config.CAM_WIDTH, 3), dtype=np.uint8)
                cv2.putText(placeholder, "CAMERA RECONNECTING...", (35, config.CAM_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(window_name, letterbox_to_screen(placeholder, screen_w, screen_h) if not args.windowed else placeholder)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            bad_frames = 0

            frame = cv2.flip(raw, 1) if config.FLIP_FRAME else raw
            h, w = frame.shape[:2]

            # Night enhancement is used by detectors; tint is display-only.
            night_active = night.update(frame)
            camera_night_mode = sync_camera_to_light_mode(cap, night_active, camera_night_mode)
            detection_frame = enhance_for_low_light(frame) if night_active else frame
            display_frame = apply_night_vision_tint(detection_frame) if (night_active and config.NIGHT_VISION_DISPLAY_TINT) else detection_frame.copy()

            # Feed only the newest frame to YOLO; old frames are dropped intentionally.
            try:
                yolo_worker.set_frame(detection_frame, night_active)
                snap = yolo_worker.snapshot()
                yolo_health = yolo_worker.health()
            except Exception as exc:
                snap = None
                yolo_health = {"ok": False, "age": float("inf"), "errors": -1, "error": str(exc)}

            # Face analysis.
            rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
            mp_image = new_mp_image(rgb)
            ts_ms = int(time.monotonic_ns() // 1_000_000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms
            face = analyze_frame(face_landmarker, mp_image, ts_ms, w, h)
            state.update_face(face.face_detected, now)

            if face.face_detected:
                fx, fy, fw, fh = face.bbox
                cv2.rectangle(display_frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)
                avg_ear = state.update_eyes(face.avg_ear, now)
                mar = state.update_yawn(face.mar, now)
                pose_ratio = state.pose_smoother.update(face.head_pose_x_ratio)
                if pose_ratio is None:
                    pose_ratio = 0.5
                if pose_ratio < config.HEAD_RIGHT_THRESHOLD:
                    head_pose = "Looking Right"
                    last_head_pose = head_pose
                    state.update_distraction(True, now)
                elif pose_ratio > config.HEAD_LEFT_THRESHOLD:
                    head_pose = "Looking Left"
                    last_head_pose = head_pose
                    state.update_distraction(True, now)
                else:
                    head_pose = "Center Focus"
                    last_head_pose = head_pose
                    state.update_distraction(False, now)
                eye_status = f"Closed EAR {avg_ear:.2f}" if avg_ear < config.EAR_THRESHOLD else f"Open EAR {avg_ear:.2f}"
                yawn_status = f"Yawning MAR {mar:.2f}" if mar > config.MAR_THRESHOLD else f"Normal MAR {mar:.2f}"
            else:
                head_pose = last_head_pose if not state.face_lost else "Face Lost"
                eye_status = "Face unavailable"
                yawn_status = "Face unavailable"
                # Do not count missing face as closed eyes or distraction.
                state.update_distraction(False, now)
                avg_ear = None
                mar = None

            # YOLO result is usable only while fresh.
            if snap is not None and not yolo_worker.is_stale():
                person_boxes = snap.person_boxes
                phone_boxes = snap.phone_boxes
                yolo_ok = True
            else:
                person_boxes = []
                phone_boxes = []
                yolo_ok = False

            driver_box = choose_driver_box(person_boxes, face.bbox if face.face_detected else None, w, driver_side)
            driver_status = "Driver Active" if driver_box else ("Passenger Present" if person_boxes else "Searching")
            if driver_box:
                x1, y1, x2, y2 = [int(v) for v in driver_box]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Phone temporal confirmation: only update on a new YOLO result.
            current_yolo_ts = 0.0 if snap is None else snap.timestamp
            if not yolo_ok or current_yolo_ts <= 0.0:
                phone_confirms = 0
                phone_misses = 0
                phone_present = False
            elif current_yolo_ts != last_yolo_ts:
                last_yolo_ts = current_yolo_ts
                candidate = any(phone_candidate(p, driver_box, w, h) for p in phone_boxes)
                if candidate:
                    phone_misses = 0
                    phone_confirms = min(config.PHONE_CONFIRMATION_LIMIT, phone_confirms + 1)
                    if phone_confirms >= config.PHONE_CONFIRMATION_LIMIT:
                        phone_present = True
                else:
                    phone_confirms = 0
                    phone_misses += 1
                    if phone_misses >= config.PHONE_CLEAR_CONFIRMATIONS:
                        phone_present = False
                for pbox in phone_boxes:
                    if driver_box and phone_candidate(pbox, driver_box, w, h):
                        cv2.rectangle(display_frame, (int(pbox[0]), int(pbox[1])), (int(pbox[2]), int(pbox[3])), (0, 0, 255), 3)

            # Hands.
            hands = analyze_hands(hand_landmarker, mp_image, ts_ms, w, h)
            driver_hands, passenger_hands = classify_driver_hands(hands, driver_box)
            state.update_hands(len(driver_hands), now)
            if driver_hands:
                draw_hands(display_frame, [p for p, _ in driver_hands], (0, 255, 0))
            if passenger_hands:
                draw_hands(display_frame, [p for p, _ in passenger_hands], (0, 165, 255))

            # Alert priority.
            active_alert = "System Nominal"
            alert_color = (0, 180, 0)
            alert_kind = None
            if state.sleep_active and phone_present:
                active_alert = "CRITICAL: DRIVER SLEEP + PHONE"
                alert_color = (0, 0, 255)
                alert_kind = "sleep_phone"
            elif state.sleep_active:
                active_alert = "CRITICAL: DRIVER SLEEP DETECTED"
                alert_color = (0, 0, 255)
                alert_kind = "sleep"
            elif phone_present:
                active_alert = "MOBILE PHONE USAGE DETECTED"
                alert_color = (0, 0, 255)
                alert_kind = "phone"
            elif state.drowsy_active:
                active_alert = "WARNING: DROWSINESS DETECTED"
                alert_color = (0, 165, 255)
                alert_kind = "drowsiness"
            elif state.yawn_active:
                active_alert = "WARNING: YAWNING DETECTED"
                alert_color = (0, 165, 255)
                alert_kind = "yawn"
            elif state.distraction_active:
                active_alert = "PLEASE FOCUS ON THE ROAD"
                alert_color = (0, 165, 255)
                alert_kind = "distraction"
            elif config.ENABLE_HAND_ALERTS and state.hands_missing:
                active_alert = "WARNING: DRIVER HANDS NOT VISIBLE"
                alert_color = (0, 165, 255)
                alert_kind = "hands"

            if alert_kind == "sleep_phone":
                sleep_and_phone_alert()
            elif alert_kind == "sleep":
                sleep_alert()
            elif alert_kind == "phone":
                phone_alert()
            elif alert_kind == "drowsiness":
                drowsiness_alert()
            elif alert_kind == "yawn":
                yawn_alert()
            elif alert_kind == "distraction":
                distraction_alert()
            elif alert_kind == "hands":
                hands_not_visible_alert()
            else:
                logger.reset()
            logger.log(driver_status, "Driver" if driver_box else "N/A", active_alert)

            # Display metrics.
            metrics = [
                f"Cabin:{driver_status}",
                f"Face:{'OK' if face.face_detected else 'NO'}",
                f"Eyes:{'Closed' if state.drowsy_active else 'Open/Scan'}",
                f"Pose:{head_pose}",
                f"Phone:{'VIOLATION' if phone_present else 'Clean'}",
                f"Yawn:{'YES' if state.yawn_active else 'NO'}",
                f"Hands:{len(driver_hands)}",
                f"Light:{night.last_luma:.0f}",
                f"YOLO:{'OK' if yolo_ok else 'WAIT'}",
            ]
            draw_zone_gridlines(display_frame, w, h, driver_side)
            draw_hud_panel(display_frame, w, metrics, night_active, yolo_ok)
            draw_alert_banner(display_frame, w, h, active_alert, alert_color)

            display = letterbox_to_screen(display_frame, screen_w, screen_h) if not args.windowed else display_frame
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_id += 1

    except KeyboardInterrupt:
        print("[INFO] Stopped by user.")
    except Exception as exc:
        print(f"[ERROR] Fatal runtime error: {exc}")
        raise
    finally:
        try:
            if yolo_worker is not None:
                yolo_worker.stop()
        except Exception:
            pass
        try:
            if face_landmarker is not None:
                face_landmarker.close()
        except Exception:
            pass
        try:
            if hand_landmarker is not None:
                hand_landmarker.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if voice_started:
            shutdown_voice()
        print(f"[INFO] Event log: {config.LOG_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
