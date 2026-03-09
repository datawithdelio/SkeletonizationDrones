import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2 as cv
import numpy as np
from ultralytics import YOLO


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    os.path.dirname(CURRENT_DIR), "utils", "models", "yolov8n-seg.onnx"
)


@dataclass
class EvalStats:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def to_metrics(self) -> Dict[str, float]:
        precision = self.tp / max(1, (self.tp + self.fp))
        recall = self.tp / max(1, (self.tp + self.fn))
        f1 = 2 * precision * recall / max(1e-12, (precision + recall))
        accuracy = (self.tp + self.tn) / max(1, (self.tp + self.tn + self.fp + self.fn))
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }


def _enhance_night_frame(image):
    ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(ycrcb)
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    enhanced = cv.merge((y, cr, cb))
    return cv.cvtColor(enhanced, cv.COLOR_YCrCb2BGR)


def _collect_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = []
    for root, _, names in os.walk(folder):
        for name in names:
            if os.path.splitext(name)[1].lower() in exts:
                files.append(os.path.join(root, name))
    return sorted(files)


def _predict_has_target(model, image, target_classes, confidence, iou):
    results = model.predict(image, conf=confidence, iou=iou, save=False, show=False, verbose=False)
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None:
            continue
        for i in range(len(boxes.cls)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
            if cls_id in target_classes and conf >= confidence:
                return True
    return False


def evaluate(
    dataset_dir: str,
    confidence: float,
    iou: float,
    target_classes: List[int],
    night_mode: bool,
    positive_dir_name: str,
    negative_dir_name: str,
    positive_label: str,
    negative_label: str,
) -> Dict:
    positive_dir = os.path.join(dataset_dir, positive_dir_name)
    negative_dir = os.path.join(dataset_dir, negative_dir_name)
    if not os.path.isdir(positive_dir) or not os.path.isdir(negative_dir):
        raise FileNotFoundError(
            f"Dataset must contain `{positive_dir_name}/` and `{negative_dir_name}/` subfolders."
        )

    model = YOLO(MODEL_PATH)
    stats = EvalStats()
    details = []

    for label_name, label_value, folder in [
        (positive_label, 1, positive_dir),
        (negative_label, 0, negative_dir),
    ]:
        for image_path in _collect_images(folder):
            image = cv.imread(image_path, cv.IMREAD_COLOR)
            if image is None:
                continue
            if night_mode:
                image = _enhance_night_frame(image)

            pred = 1 if _predict_has_target(model, image, target_classes, confidence, iou) else 0
            details.append(
                {
                    "image": image_path,
                    "label": label_name,
                    "pred": positive_label if pred else negative_label,
                }
            )

            if label_value == 1 and pred == 1:
                stats.tp += 1
            elif label_value == 1 and pred == 0:
                stats.fn += 1
            elif label_value == 0 and pred == 1:
                stats.fp += 1
            else:
                stats.tn += 1

    report = {
        "dataset_dir": dataset_dir,
        "confidence": confidence,
        "iou": iou,
        "target_classes": target_classes,
        "night_mode": night_mode,
        "confusion_matrix": {
            "tp": stats.tp,
            "tn": stats.tn,
            "fp": stats.fp,
            "fn": stats.fn,
        },
        "metrics": stats.to_metrics(),
        "samples_evaluated": len(details),
        "details": details,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary object discrimination (e.g., cars/noncars, drones/birds).")
    parser.add_argument("--dataset-dir", required=True, help="Path containing positive/negative class folders.")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--target-classes", default="4", help="Comma-separated YOLO class IDs treated as positive.")
    parser.add_argument("--night-mode", action="store_true", help="Apply CLAHE-based night enhancement.")
    parser.add_argument("--positive-dir-name", default="drones")
    parser.add_argument("--negative-dir-name", default="birds")
    parser.add_argument("--positive-label", default="drone")
    parser.add_argument("--negative-label", default="bird")
    parser.add_argument("--output", default="binary_eval_report.json")
    args = parser.parse_args()

    target_classes = [int(x.strip()) for x in args.target_classes.split(",") if x.strip()]
    report = evaluate(
        dataset_dir=args.dataset_dir,
        confidence=args.confidence,
        iou=args.iou,
        target_classes=target_classes,
        night_mode=args.night_mode,
        positive_dir_name=args.positive_dir_name,
        negative_dir_name=args.negative_dir_name,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report["metrics"], indent=2))
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
