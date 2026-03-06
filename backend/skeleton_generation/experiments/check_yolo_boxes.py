import argparse
import json
import os

import cv2 as cv
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    os.path.dirname(CURRENT_DIR), "utils", "models", "yolov8n-seg.onnx"
)


def inspect_boxes(image_path, confidence=0.25, iou=0.65):
    image = cv.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    model = YOLO(MODEL_PATH)
    results = model.predict(image, conf=confidence, iou=iou, save=False, show=False, verbose=False)
    detections = []

    for result in results or []:
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.xyxy is None:
            continue
        for i in range(len(boxes.xyxy)):
            xyxy = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
            cls = int(boxes.cls[i]) if boxes.cls is not None else -1
            detections.append(
                {
                    "box_xyxy": [round(v, 2) for v in xyxy],
                    "confidence": round(conf, 4),
                    "class_id": cls,
                }
            )

    return {
        "image_path": image_path,
        "detections_count": len(detections),
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect YOLO box outputs on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.65)
    args = parser.parse_args()

    report = inspect_boxes(args.image, confidence=args.confidence, iou=args.iou)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
