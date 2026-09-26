"""Single-window dashboard drawing."""
import cv2
import numpy as np
import config


def _fit_text(text, max_width, scale=0.42, thickness=1):
    text = str(text)
    while True:
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if w <= max_width:
            return text
        if len(text) <= 8:
            return text
        text = text[:-4] + "..."


def draw_zone_gridlines(frame, w, h, driver_side):
    cx = w // 2
    cv2.line(frame, (cx, 0), (cx, h), (0, 200, 255), 1)
    color = (0, 220, 255)
    label_y = config.HUD_Y_OFFSET + config.HUD_HEIGHT + 22
    if driver_side == "right":
        cv2.putText(frame, "PASSENGER ZONE", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(frame, "DRIVER ZONE", (cx + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "DRIVER ZONE", (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(frame, "PASSENGER ZONE", (cx + 10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_hud_panel(frame, w, metrics, night_active, yolo_ok):
    x = config.HUD_X_OFFSET
    y = config.HUD_Y_OFFSET
    hud_w = max(200, w - 2 * x)
    hud_h = config.HUD_HEIGHT
    bg = (35, 15, 15) if night_active else (35, 35, 35)
    border = (0, 160, 0) if night_active else (120, 120, 120)
    cv2.rectangle(frame, (x, y), (x + hud_w, y + hud_h), bg, cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + hud_w, y + hud_h), border, 2)
    left = " | ".join(metrics[:5])
    right = " | ".join(metrics[5:])
    max_width = hud_w - 20
    cv2.putText(frame, _fit_text(left, max_width), (x + 10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, _fit_text(right, max_width), (x + 10, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    status = "YOLO OK" if yolo_ok else "YOLO DEGRADED"
    status_color = (0, 220, 0) if yolo_ok else (0, 0, 255)
    cv2.putText(frame, status, (w - 120, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, status_color, 1, cv2.LINE_AA)
    if night_active:
        cv2.putText(frame, "NIGHT VISION", (w - 120, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 220, 0), 1, cv2.LINE_AA)


def draw_alert_banner(frame, w, h, alert, color):
    y1 = max(10, h - config.ALERT_BANNER_HEIGHT - 8)
    y2 = h - 8
    cv2.rectangle(frame, (10, y1), (w - 10, y2), color, cv2.FILLED)
    max_width = w - 40
    text = _fit_text(alert, max_width, 0.65, 2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(frame, text, (max(20, (w - tw) // 2), y1 + (y2 - y1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def letterbox_to_screen(frame, screen_w, screen_h):
    h, w = frame.shape[:2]
    scale = min(screen_w / max(1, w), screen_h / max(1, h))
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=frame.dtype)
    y = (screen_h - nh) // 2
    x = (screen_w - nw) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def get_screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width, height = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return int(width), int(height)
    except Exception:
        return 1920, 1080
