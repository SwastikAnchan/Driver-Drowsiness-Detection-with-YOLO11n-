"""
audio_alert.py
--------------
Voice-only alert system for the hardened driver monitoring system.

Windows / SAPI5:
- Creates a fresh pyttsx3 engine for every utterance.
- Does not reuse a long-lived SAPI5 engine.
- No audible beep is generated.
- Keeps compatibility functions used by older/newer main.py versions.
"""

import threading
import time

import config

_lock = threading.Lock()
_voice_lock = threading.Lock()
_last_voice_times = {}


def _speak_now(message):
    """Speak one message using a fresh SAPI5 engine."""
    if not getattr(config, "VOICE_ENABLED", True):
        return False

    try:
        import pyttsx3

        # A fresh engine per utterance avoids the common Windows/SAPI5
        # failure where a reused engine speaks once and then goes silent.
        with _voice_lock:
            engine = pyttsx3.init("sapi5")
            engine.setProperty("rate", getattr(config, "VOICE_RATE", 165))
            engine.setProperty("volume", getattr(config, "VOICE_VOLUME", 1.0))

            engine.say(message)
            engine.runAndWait()

            try:
                engine.stop()
            except Exception:
                pass

        print(f"[VOICE] Spoken: {message}")
        return True

    except Exception as exc:
        print(f"[VOICE ERROR] Speech failed: {exc}")
        return False


def _speak_worker(message):
    try:
        _speak_now(message)
    except Exception as exc:
        print(f"[VOICE ERROR] Worker failed: {exc}")


def voice_alert(message, alert_key="general", cooldown=None):
    """Speak a non-blocking voice alert with a per-alert cooldown."""
    if not getattr(config, "VOICE_ENABLED", True):
        return False

    if cooldown is None:
        cooldown = getattr(config, "VOICE_ALERT_COOLDOWN_SEC", 8.0)

    now = time.monotonic()

    with _lock:
        last_time = _last_voice_times.get(alert_key, 0.0)
        if now - last_time < cooldown:
            return False
        _last_voice_times[alert_key] = now

    threading.Thread(
        target=_speak_worker,
        args=(message,),
        name="VoiceAlert",
        daemon=True,
    ).start()

    return True


def alert_beep(frequency=0, duration=0):
    """Compatibility no-op. The project is intentionally voice-only."""
    return False


def sleep_alert():
    return voice_alert(
        "Warning! Driver sleep detected. Please wake up and focus on the road.",
        alert_key="sleep",
    )


def phone_alert():
    return voice_alert(
        "Warning! Mobile phone usage detected. Please put your phone away and focus on driving.",
        alert_key="phone",
    )


def sleep_and_phone_alert():
    return voice_alert(
        "Critical warning! Driver sleep and mobile phone usage detected. Please put the phone away, wake up, and focus on the road.",
        alert_key="sleep_and_phone",
        cooldown=6.0,
    )


def drowsiness_alert():
    return voice_alert(
        "Warning! Driver drowsiness detected. Please stay alert and focus on the road.",
        alert_key="drowsiness",
    )


def distraction_alert():
    return voice_alert(
        "Warning! Please focus on the road.",
        alert_key="distraction",
    )


def yawn_alert():
    return voice_alert(
        "Warning! Driver yawning detected. Please stay alert.",
        alert_key="yawn",
    )


def hands_off_wheel_alert():
    return voice_alert(
        "Warning! Please keep your hands on the steering wheel.",
        alert_key="hands_off_wheel",
    )


def hands_not_visible_alert():
    return voice_alert(
        "Warning! Driver hands are not visible. Please keep your hands on the steering wheel.",
        alert_key="hands_not_visible",
    )


def shutdown_voice():
    """Compatibility shutdown hook used by the hardened main.py."""
    # There is no persistent engine to stop. Current speech workers are
    # daemon threads and each owns its own temporary engine instance.
    return None


def stop():
    """Compatibility alias for older project versions."""
    return shutdown_voice()
