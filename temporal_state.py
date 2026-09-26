"""FPS-independent temporal state and smoothing."""
from collections import deque
import time

import config


class FeatureSmoother:
    def __init__(self, size):
        self.values = deque(maxlen=max(1, int(size)))

    def update(self, value):
        if value is None:
            return None
        self.values.append(float(value))
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values.clear()


class DriverState:
    def __init__(self):
        self.eye_closed_since = None
        self.eye_recovery_since = None
        self.yawn_since = None
        self.yawn_recovery_since = None
        self.distraction_since = None
        self.distraction_recovery_since = None
        self.face_lost_since = None
        self.hands_missing_since = None
        self.hands_recovery_since = None
        self.eye_smoother = FeatureSmoother(config.FEATURE_SMOOTHING_SAMPLES)
        self.mar_smoother = FeatureSmoother(config.FEATURE_SMOOTHING_SAMPLES)
        self.pose_smoother = FeatureSmoother(config.FEATURE_SMOOTHING_SAMPLES)

        self.sleep_active = False
        self.drowsy_active = False
        self.yawn_active = False
        self.distraction_active = False
        self.face_lost = False
        self.hands_missing = False

    @staticmethod
    def _elapsed(start, now):
        return 0.0 if start is None else max(0.0, now - start)

    @staticmethod
    def _update_condition(flag, since, recovery_since, active, now, trigger_sec, recovery_sec):
        if flag:
            if since is None:
                since = now
            recovery_since = None
            active = active or (now - since >= trigger_sec)
        else:
            since = None
            if active:
                if recovery_since is None:
                    recovery_since = now
                if now - recovery_since >= recovery_sec:
                    active = False
                    recovery_since = None
            else:
                recovery_since = None
        return since, recovery_since, active

    def update_face(self, face_detected, now=None):
        now = time.monotonic() if now is None else now
        if face_detected:
            self.face_lost_since = None
            self.face_lost = False
        else:
            if self.face_lost_since is None:
                self.face_lost_since = now
            self.face_lost = (now - self.face_lost_since) >= config.FACE_GRACE_SEC

    def update_eyes(self, avg_ear, now=None):
        now = time.monotonic() if now is None else now
        smoothed = self.eye_smoother.update(avg_ear)
        closed = smoothed is not None and smoothed < config.EAR_THRESHOLD
        self.eye_closed_since, self.eye_recovery_since, self.sleep_active = self._update_condition(
            closed, self.eye_closed_since, self.eye_recovery_since, self.sleep_active, now,
            config.SLEEP_DURATION_SEC, config.RECOVERY_DURATION_SEC,
        )
        # Drowsiness is a shorter version of the same eye-closure condition.
        if closed:
            if self._elapsed(self.eye_closed_since, now) >= config.DROWSY_DURATION_SEC:
                self.drowsy_active = True
        else:
            self.drowsy_active = False
        return smoothed

    def update_yawn(self, mar, now=None):
        now = time.monotonic() if now is None else now
        smoothed = self.mar_smoother.update(mar)
        yawn = smoothed is not None and smoothed > config.MAR_THRESHOLD
        self.yawn_since, self.yawn_recovery_since, self.yawn_active = self._update_condition(
            yawn, self.yawn_since, self.yawn_recovery_since, self.yawn_active, now,
            config.YAWN_DURATION_SEC, config.RECOVERY_DURATION_SEC,
        )
        return smoothed

    def update_distraction(self, looking_away, now=None):
        now = time.monotonic() if now is None else now
        self.distraction_since, self.distraction_recovery_since, self.distraction_active = self._update_condition(
            looking_away, self.distraction_since, self.distraction_recovery_since, self.distraction_active, now,
            config.DISTRACTION_DURATION_SEC, config.RECOVERY_DURATION_SEC,
        )

    def update_hands(self, hand_count, now=None):
        now = time.monotonic() if now is None else now
        missing = hand_count <= 0
        self.hands_missing_since, self.hands_recovery_since, self.hands_missing = self._update_condition(
            missing, self.hands_missing_since, self.hands_recovery_since, self.hands_missing, now,
            config.HANDS_NOT_VISIBLE_SEC, config.RECOVERY_DURATION_SEC,
        )

    def reset_for_camera_reconnect(self):
        self.__init__()
