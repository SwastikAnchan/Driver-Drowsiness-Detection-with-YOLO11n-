"""Simple one-window camera diagnostic preview."""
import argparse
import cv2
from camera_utils import open_camera


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--backend", choices=["dshow", "msmf", "default"], default=None)
    args = parser.parse_args()
    cap = open_camera(args.camera_index, args.backend)
    if cap is None:
        return 1
    try:
        cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)
        while True:
            ret, frame = cap.read()
            if ret and frame is not None and frame.size:
                cv2.imshow("Camera Preview", frame)
            else:
                canvas = 255 * (frame[:480, :640] * 0 if frame is not None else 0)
                cv2.imshow("Camera Preview", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
