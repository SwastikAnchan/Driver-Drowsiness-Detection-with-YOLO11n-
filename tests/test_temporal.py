import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from temporal_state import DriverState
import config


def test_sleep_is_time_based():
    s = DriverState()
    now = 100.0
    s.update_face(True, now)
    s.eye_smoother.update(config.EAR_THRESHOLD - 0.05)
    s.update_eyes(config.EAR_THRESHOLD - 0.05, now)
    assert not s.sleep_active
    s.update_eyes(config.EAR_THRESHOLD - 0.05, now + config.SLEEP_DURATION_SEC + 0.01)
    assert s.sleep_active
