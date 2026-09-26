"""Check camera indices/backends without loading AI models."""
import platform
import time
import cv2


def main():
    if platform.system() == "Windows":
        backends = [("dshow", cv2.CAP_DSHOW), ("msmf", cv2.CAP_MSMF), ("default", None)]
    else:
        backends = [("default", None)]
    found = 0
    for tag, backend in backends:
        for index in range(5):
            cap = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            valid = False
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.size:
                    valid = True
                    break
                time.sleep(0.05)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            print(f"index={index} backend={tag} opened=True frame={valid} size={width}x{height} fps={fps:.1f}")
            if valid:
                found += 1
            cap.release()
    print(f"Working camera combinations: {found}")
    return 0 if found else 1


if __name__ == "__main__":
    raise SystemExit(main())
