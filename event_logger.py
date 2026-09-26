"""Safe CSV event logger."""
import csv
from datetime import datetime
from pathlib import Path
import threading


class EventLogger:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self._last_event = ""
        self._lock = threading.Lock()
        self.last_error = None

    def reset(self):
        with self._lock:
            self._last_event = ""

    def log(self, cabin, identity, event):
        if not event or event == "System Nominal":
            return True
        with self._lock:
            if event == self._last_event:
                return True
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                new_file = not self.log_file.exists() or self.log_file.stat().st_size == 0
                with self.log_file.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    if new_file:
                        writer.writerow(["timestamp", "cabin", "identity", "event"])
                    writer.writerow([
                        datetime.now().astimezone().isoformat(timespec="seconds"),
                        cabin,
                        identity,
                        event,
                    ])
                    handle.flush()
                self._last_event = event
                self.last_error = None
                return True
            except OSError as exc:
                self.last_error = str(exc)
                print(f"[WARN] Event log write failed: {exc}")
                # Do not crash the monitoring system because a log file is locked/full.
                return False
