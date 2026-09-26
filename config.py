"""Central configuration for the driver monitoring system."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
YOLO_MODEL_PATH = MODELS_DIR / "yolo11n.pt"
YOLO_MODEL_NAME = "yolo11n.pt"
LOG_FILE = BASE_DIR / "event_history_log.csv"

# ---------------- Camera ----------------
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30
CAMERA_SCAN_MAX_INDEX = 4
CAMERA_BAD_FRAME_LIMIT = 12
CAMERA_RECONNECT_RETRIES = 4
CAMERA_OPEN_DELAY_SEC = 0.25

# ---------------- Display ----------------
WINDOW_NAME = "Driver Monitoring System"
FULLSCREEN = True
FLIP_FRAME = True
DRIVER_SIDE = "right"  # India / RHD

# ---------------- YOLO ----------------
YOLO_INFER_SCALE = 1.0
YOLO_INTERVAL_SEC = 0.08
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = None  # None = Ultralytics default device selection
YOLO_PERSON_CONF = 0.45
YOLO_PHONE_CONF = 0.25
YOLO_PERSON_CONF_NIGHT = 0.40
YOLO_PHONE_CONF_NIGHT = 0.22
YOLO_STALE_AFTER_SEC = 0.90
YOLO_ERROR_LOG_INTERVAL_SEC = 5.0

# COCO labels used by the pretrained detector.
COCO_PERSON_CLASS = 0
COCO_CELL_PHONE_CLASS = 67

# Phone confirmation is based on NEW YOLO results, not camera frames.
PHONE_CONTAINMENT_THRESHOLD = 0.30
PHONE_CENTER_INSIDE_THRESHOLD = 0.10
PHONE_CONFIRMATION_LIMIT = 3
PHONE_CLEAR_CONFIRMATIONS = 2
PHONE_RESULT_TIMEOUT_SEC = 1.10
DRIVER_BOX_EXPANSION = 0.08

# ---------------- Face / drowsiness ----------------
EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.55
HEAD_LEFT_THRESHOLD = 0.62
HEAD_RIGHT_THRESHOLD = 0.38
FEATURE_SMOOTHING_SAMPLES = 5

# These preserve the old ~30-FPS behavior while making it FPS independent.
DROWSY_DURATION_SEC = 20.0 / CAM_FPS
SLEEP_DURATION_SEC = 45.0 / CAM_FPS
YAWN_DURATION_SEC = 20.0 / CAM_FPS
DISTRACTION_DURATION_SEC = 30.0 / CAM_FPS
RECOVERY_DURATION_SEC = 0.15
FACE_GRACE_SEC = 0.20

# ---------------- Hands ----------------
HAND_MIN_DETECTION_CONF = 0.50
HAND_MIN_TRACKING_CONF = 0.50
HAND_MAX_COUNT = 2
ENABLE_HAND_ALERTS = False
HANDS_NOT_VISIBLE_SEC = 2.0

# ---------------- Night vision ----------------
ENABLE_NIGHT_VISION = True
NIGHT_MODE_ENTER_LUMA = 70.0
NIGHT_MODE_EXIT_LUMA = 85.0
BRIGHTNESS_SAMPLE_DOWNSCALE = 0.25
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID = (8, 8)
GAMMA_NIGHT = 1.8
DENOISE_H_NIGHT = 3
SHARPEN_NIGHT = True
NIGHT_VISION_DISPLAY_TINT = True
NIGHT_CAM_GAIN = None
NIGHT_CAM_EXPOSURE = None
DAY_CAM_GAIN = None
DAY_CAM_EXPOSURE = None

# ---------------- Voice ----------------
VOICE_ENABLED = True
VOICE_RATE = 165
VOICE_VOLUME = 1.0
VOICE_QUEUE_SIZE = 3
VOICE_DEFAULT_COOLDOWN_SEC = 4.0

VOICE_COOLDOWNS = {
    "sleep": 4.0,
    "sleep_phone": 4.0,
    "phone": 4.0,
    "drowsiness": 5.0,
    "yawn": 6.0,
    "distraction": 5.0,
    "hands": 6.0,
}

# ---------------- Model download ----------------
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
AUTO_DOWNLOAD_FACE_MODEL = True
AUTO_DOWNLOAD_YOLO_MODEL = True

# ---------------- Face landmark indices ----------------
LEFT_EYE_EAR_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_EAR_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_TOP_IDX = [82, 13, 312]
MOUTH_BOTTOM_IDX = [87, 14, 317]
MOUTH_LEFT_IDX = 78
MOUTH_RIGHT_IDX = 308
NOSE_TIP_IDX = 1

# ---------------- HUD ----------------
HUD_HEIGHT = 58
HUD_X_OFFSET = 10
HUD_Y_OFFSET = 10
ALERT_BANNER_HEIGHT = 54

# ---------------- Evaluation target ----------------
TARGET_ACCURACY = 0.90
TARGET_ACCURACY_HIGH = 0.95
