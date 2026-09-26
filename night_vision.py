"""Low-light enhancement with stable day/night hysteresis."""
import cv2
import numpy as np
import config


def frame_mean_luma(frame_bgr):
    small = cv2.resize(frame_bgr, None, fx=config.BRIGHTNESS_SAMPLE_DOWNSCALE,
                       fy=config.BRIGHTNESS_SAMPLE_DOWNSCALE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def _gamma_correct(gray_u8, gamma):
    gamma = max(0.1, float(gamma))
    inv = 1.0 / gamma
    table = np.clip(((np.arange(256) / 255.0) ** inv) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(gray_u8, table)


class NightModeState:
    def __init__(self):
        self.active = False
        self.last_luma = 255.0

    def update(self, frame_bgr):
        self.last_luma = frame_mean_luma(frame_bgr)
        if not config.ENABLE_NIGHT_VISION:
            self.active = False
        elif not self.active and self.last_luma < config.NIGHT_MODE_ENTER_LUMA:
            self.active = True
        elif self.active and self.last_luma > config.NIGHT_MODE_EXIT_LUMA:
            self.active = False
        return self.active


def enhance_for_low_light(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                            tileGridSize=config.CLAHE_TILE_GRID)
    l = clahe.apply(l)
    l = _gamma_correct(l, config.GAMMA_NIGHT)
    if config.DENOISE_H_NIGHT > 0:
        # Mild denoising keeps CPU cost reasonable at 640x480.
        l = cv2.fastNlMeansDenoising(l, None, h=config.DENOISE_H_NIGHT,
                                     templateWindowSize=7, searchWindowSize=15)
    if config.SHARPEN_NIGHT:
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=1.2)
        l = cv2.addWeighted(l, 1.35, blur, -0.35, 0)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def apply_night_vision_tint(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    tinted = np.zeros_like(frame_bgr)
    tinted[:, :, 1] = gray
    return tinted


def apply_night_camera_settings(cap):
    if config.NIGHT_CAM_GAIN is not None:
        try:
            cap.set(cv2.CAP_PROP_GAIN, config.NIGHT_CAM_GAIN)
        except Exception:
            pass
    if config.NIGHT_CAM_EXPOSURE is not None:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, config.NIGHT_CAM_EXPOSURE)
        except Exception:
            pass


def apply_day_camera_settings(cap):
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    except Exception:
        pass
    if config.DAY_CAM_GAIN is not None:
        try:
            cap.set(cv2.CAP_PROP_GAIN, config.DAY_CAM_GAIN)
        except Exception:
            pass
