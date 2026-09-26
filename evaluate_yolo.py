"""Evaluate a trained YOLO model and enforce a measurable quality gate."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/data.yaml")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--min-map50", type=float, default=0.90)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    results = YOLO(str(model_path)).val(data=args.data, imgsz=args.imgsz, device=args.device, verbose=False)
    map50 = float(getattr(results.box, "map50", 0.0))
    map5095 = float(getattr(results.box, "map", 0.0))
    precision = float(getattr(results.box, "mp", 0.0))
    recall = float(getattr(results.box, "mr", 0.0))
    print(f"mAP@0.50:     {map50:.4f}")
    print(f"mAP@0.50:0.95:{map5095:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    if map50 < args.min_map50:
        print(f"QUALITY GATE: FAIL (mAP50 < {args.min_map50:.2f})")
        return 2
    print(f"QUALITY GATE: PASS (mAP50 >= {args.min_map50:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
