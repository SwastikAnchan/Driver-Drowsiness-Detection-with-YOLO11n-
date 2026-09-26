"""Reliable voice-only alert service.
Windows-safe pyttsx3 recovery.
No beep or audio-tone code.
"""

import queue
import threading
import time

import config


class VoiceAlertService:

    def __init__(self):

        self._queue = queue.Queue(maxsize=3)

        self._cooldown = {}
        self._cooldown_lock = threading.Lock()

        self._stop = threading.Event()

        self._engine = None
        self._engine_lock = threading.Lock()

        self.available = bool(config.VOICE_ENABLED)

        self._thread = threading.Thread(
            target=self._worker,
            name="VoiceAlert",
            daemon=True,
        )

        self._thread.start()

    # ---------------------------------------------------------
    # CREATE / RECREATE VOICE ENGINE
    # ---------------------------------------------------------

    def _get_engine(self):

        if not self.available:
            return None

        with self._engine_lock:

            if self._engine is not None:
                return self._engine

            try:

                import pyttsx3

                print("[VOICE] Initializing voice engine...")

                engine = pyttsx3.init("sapi5")

                engine.setProperty(
                    "rate",
                    config.VOICE_RATE
                )

                engine.setProperty(
                    "volume",
                    config.VOICE_VOLUME
                )

                self._engine = engine

                print("[VOICE] Voice engine ready.")

                return engine

            except Exception as exc:

                print(
                    f"[VOICE ERROR] Could not initialize voice engine: {exc}"
                )

                self._engine = None

                return None

    # ---------------------------------------------------------
    # DESTROY BROKEN ENGINE
    # ---------------------------------------------------------

    def _reset_engine(self):

        with self._engine_lock:

            if self._engine is not None:

                try:
                    self._engine.stop()
                except Exception:
                    pass

            self._engine = None

        print("[VOICE] Voice engine reset.")

    # ---------------------------------------------------------
    # SPEAK
    # ---------------------------------------------------------

    def _speak_now(self, message):

        engine = self._get_engine()

        if engine is None:
            return False

        try:

            with self._engine_lock:

                engine.say(message)
                engine.runAndWait()

            print(f"[VOICE] Spoken: {message}")

            return True

        except Exception as exc:

            print(
                f"[VOICE ERROR] Speech failed: {exc}"
            )

            # VERY IMPORTANT:
            # pyttsx3/SAPI may become unusable.
            # Throw away the broken engine so the next
            # alert gets a fresh engine.
            self._reset_engine()

            return False

    # ---------------------------------------------------------
    # WORKER
    # ---------------------------------------------------------

    def _worker(self):

        while not self._stop.is_set():

            try:

                message, key = self._queue.get(
                    timeout=0.2
                )

            except queue.Empty:

                continue

            if message is None:

                self._queue.task_done()
                break

            try:

                success = self._speak_now(message)

                if not success:
                    print(
                        f"[VOICE] Failed alert: {key}"
                    )

            finally:

                self._queue.task_done()

    # ---------------------------------------------------------
    # PUBLIC SPEAK
    # ---------------------------------------------------------

    def speak(
        self,
        message,
        key="general",
        cooldown=None,
    ):

        if not self.available:
            return False

        if not message:
            return False

        if cooldown is None:
            cooldown = config.VOICE_DEFAULT_COOLDOWN_SEC

        now = time.monotonic()

        # Check cooldown.
        with self._cooldown_lock:

            last = self._cooldown.get(
                key,
                0.0
            )

            if now - last < cooldown:

                return False

        # Put message in queue.
        try:

            self._queue.put_nowait(
                (message, key)
            )

        except queue.Full:

            print(
                f"[VOICE] Queue full: {key}"
            )

            return False

        # IMPORTANT:
        # Only consume cooldown after queue insertion.
        with self._cooldown_lock:

            self._cooldown[key] = now

        print(
            f"[VOICE] Queued: {key}"
        )

        return True

    # ---------------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------------

    def stop(self):

        print("[VOICE] Shutting down...")

        self._stop.set()

        try:

            while True:

                self._queue.get_nowait()
                self._queue.task_done()

        except queue.Empty:

            pass

        try:

            self._queue.put_nowait(
                (None, "shutdown")
            )

        except queue.Full:

            pass

        self._thread.join(
            timeout=3.0
        )

        self._reset_engine()

        print("[VOICE] Voice service stopped.")


# =============================================================
# GLOBAL VOICE SERVICE
# =============================================================

_voice = VoiceAlertService()


def _say(message, key):

    cooldown = config.VOICE_COOLDOWNS.get(
        key,
        config.VOICE_DEFAULT_COOLDOWN_SEC,
    )

    return _voice.speak(
        message,
        key=key,
        cooldown=cooldown,
    )


# =============================================================
# ALERTS
# =============================================================

def sleep_alert():

    return _say(
        "Critical warning. Driver sleep detected. "
        "Wake up and focus on the road.",
        "sleep",
    )


def phone_alert():

    return _say(
        "Warning. Mobile phone usage detected. "
        "Put the phone away and focus on driving.",
        "phone",
    )


def sleep_and_phone_alert():

    return _say(
        "Critical warning. Sleep and mobile phone usage detected. "
        "Put the phone away, wake up, and focus on driving.",
        "sleep_phone",
    )


def drowsiness_alert():

    return _say(
        "Warning. Driver drowsiness detected. "
        "Please stay alert and focus on driving.",
        "drowsiness",
    )


def yawn_alert():

    return _say(
        "Warning. Driver yawning detected. "
        "Please stay alert.",
        "yawn",
    )


def distraction_alert():

    return _say(
        "Warning. Please focus on the road.",
        "distraction",
    )


def hands_not_visible_alert():

    return _say(
        "Warning. Driver hands are not visible. "
        "Keep your hands ready to control the vehicle.",
        "hands",
    )


def shutdown_voice():

    _voice.stop()