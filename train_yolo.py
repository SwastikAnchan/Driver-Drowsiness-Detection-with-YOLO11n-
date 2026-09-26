"""Optional custom YOLO11 training entry point.

Put a standard Ultralytics detection dataset under dataset/ and edit
 dataset/data.yaml. This is the supported route for targeting a measured
90-95%% event-detection score on your actual cabin camera conditions.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1, ...")
    args = parser.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    model = YOLO(args.model)
    results = model.train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, device=args.device)
    print(results)
    print("Training complete. Copy the best.pt checkpoint into models/yolo11n.pt or update config.py.")


if __name__ == "__main__":
    main()
