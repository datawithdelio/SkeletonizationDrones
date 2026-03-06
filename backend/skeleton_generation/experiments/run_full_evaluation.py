import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2 as cv
from ultralytics import YOLO

from skeleton_generation.experiments.check_yolo_boxes import MODEL_PATH
from skeleton_generation.experiments.evaluate_drone_vs_bird import _collect_images, _enhance_night_frame
from skeleton_generation.experiments.benchmark_skeleton_methods import benchmark as run_benchmark


@dataclass
class Confusion:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    def update(self, label: int, pred: int):
        if label == 1 and pred == 1:
            self.tp += 1
        elif label == 1 and pred == 0:
            self.fn += 1
        elif label == 0 and pred == 1:
            self.fp += 1
        else:
            self.tn += 1

    def metrics(self) -> Dict[str, float]:
        precision = self.tp / max(1, (self.tp + self.fp))
        recall = self.tp / max(1, (self.tp + self.fn))
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        accuracy = (self.tp + self.tn) / max(1, self.tp + self.tn + self.fp + self.fn)
        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
        }


def _predict(model, image, target_classes, confidence, iou):
    t0 = time.perf_counter()
    results = model.predict(image, conf=confidence, iou=iou, save=False, show=False, verbose=False)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    best_conf = 0.0
    detected = False
    boxes_count = 0
    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.cls is None:
            continue
        boxes_count += len(boxes.cls)
        for idx in range(len(boxes.cls)):
            cls_id = int(boxes.cls[idx])
            conf = float(boxes.conf[idx]) if boxes.conf is not None else 0.0
            if cls_id in target_classes and conf >= confidence:
                detected = True
                best_conf = max(best_conf, conf)

    return int(detected), round(latency_ms, 3), round(best_conf, 4), int(boxes_count)


def _iter_splits(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """
    Returns list of (split_name, drones_dir, birds_dir)

    Supports:
    1) dataset/day/{drones,birds} and dataset/night/{drones,birds}
    2) dataset/{drones,birds} as single split named "all"
    """
    day_d = os.path.join(dataset_dir, "day", "drones")
    day_b = os.path.join(dataset_dir, "day", "birds")
    night_d = os.path.join(dataset_dir, "night", "drones")
    night_b = os.path.join(dataset_dir, "night", "birds")

    splits = []
    if os.path.isdir(day_d) and os.path.isdir(day_b):
        splits.append(("day", day_d, day_b))
    if os.path.isdir(night_d) and os.path.isdir(night_b):
        splits.append(("night", night_d, night_b))

    if splits:
        return splits

    root_d = os.path.join(dataset_dir, "drones")
    root_b = os.path.join(dataset_dir, "birds")
    if os.path.isdir(root_d) and os.path.isdir(root_b):
        return [("all", root_d, root_b)]

    raise FileNotFoundError(
        "Dataset layout not found. Expected either day/night splits or root drones/birds folders."
    )


def run_evaluation(
    dataset_dir: str,
    output_dir: str,
    confidence: float,
    iou: float,
    target_classes: List[int],
    benchmark_image: str = "",
):
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(MODEL_PATH)
    report = {
        "dataset_dir": dataset_dir,
        "target_classes": target_classes,
        "confidence": confidence,
        "iou": iou,
        "splits": {},
        "aggregate": {},
        "method_comparison_table": None,
    }

    agg_conf = Confusion()
    agg_latencies = []
    agg_detections_conf = []

    for split_name, drones_dir, birds_dir in _iter_splits(dataset_dir):
        split_conf = Confusion()
        split_lat = []
        split_best_conf = []
        sample_rows = []

        for label_name, label_value, folder in [("drone", 1, drones_dir), ("bird", 0, birds_dir)]:
            for image_path in _collect_images(folder):
                image = cv.imread(image_path, cv.IMREAD_COLOR)
                if image is None:
                    continue

                # At night split, also test enhanced frame for robustness.
                if split_name == "night":
                    image = _enhance_night_frame(image)

                pred, latency_ms, best_conf, boxes_count = _predict(
                    model,
                    image,
                    target_classes=target_classes,
                    confidence=confidence,
                    iou=iou,
                )

                split_conf.update(label_value, pred)
                agg_conf.update(label_value, pred)
                split_lat.append(latency_ms)
                agg_latencies.append(latency_ms)
                if best_conf > 0:
                    split_best_conf.append(best_conf)
                    agg_detections_conf.append(best_conf)

                sample_rows.append(
                    {
                        "image": image_path,
                        "label": label_name,
                        "prediction": "drone" if pred else "bird",
                        "latency_ms": latency_ms,
                        "best_confidence": best_conf,
                        "boxes_count": boxes_count,
                    }
                )

        report["splits"][split_name] = {
            "samples": len(sample_rows),
            "confusion_matrix": vars(split_conf),
            "metrics": split_conf.metrics(),
            "latency_ms": {
                "mean": round(sum(split_lat) / max(1, len(split_lat)), 3),
                "p95": round(sorted(split_lat)[int(0.95 * (len(split_lat) - 1))], 3) if split_lat else 0.0,
            },
            "detection_confidence": {
                "mean": round(sum(split_best_conf) / max(1, len(split_best_conf)), 4),
                "count": len(split_best_conf),
            },
            "details": sample_rows,
        }

    report["aggregate"] = {
        "samples": sum(v["samples"] for v in report["splits"].values()),
        "confusion_matrix": vars(agg_conf),
        "metrics": agg_conf.metrics(),
        "latency_ms": {
            "mean": round(sum(agg_latencies) / max(1, len(agg_latencies)), 3),
            "p95": round(sorted(agg_latencies)[int(0.95 * (len(agg_latencies) - 1))], 3) if agg_latencies else 0.0,
        },
        "detection_confidence": {
            "mean": round(sum(agg_detections_conf) / max(1, len(agg_detections_conf)), 4),
            "count": len(agg_detections_conf),
        },
    }

    # Optional: run method comparison benchmark on one sample image.
    if benchmark_image:
        method_out = os.path.join(output_dir, "method_benchmark")
        report_path = run_benchmark(
            image_path=benchmark_image,
            output_dir=method_out,
            confidence=confidence,
            iou=iou,
        )
        with open(report_path, "r", encoding="utf-8") as f:
            method_report = json.load(f)
        report["method_comparison_table"] = method_report.get("methods", [])

    out_json = os.path.join(output_dir, "strict_evaluation_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return out_json


def main():
    parser = argparse.ArgumentParser(description="Strict evaluation runner for day/night drone-vs-bird + latency.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="evaluation_outputs")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--target-classes", default="4")
    parser.add_argument("--benchmark-image", default="", help="Optional image path for method comparison table.")
    args = parser.parse_args()

    target_classes = [int(x.strip()) for x in args.target_classes.split(",") if x.strip()]
    out_json = run_evaluation(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        confidence=args.confidence,
        iou=args.iou,
        target_classes=target_classes,
        benchmark_image=args.benchmark_image,
    )
    print(f"Saved strict evaluation report: {out_json}")


if __name__ == "__main__":
    main()
