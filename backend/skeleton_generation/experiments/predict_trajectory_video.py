import argparse
import csv
import time

import cv2 as cv
from ultralytics import YOLO

from skeleton_generation.experiments.check_yolo_boxes import MODEL_PATH
from skeleton_generation.utils.tracking.trajectory import (
    KalmanTrajectoryPredictor,
    TrajectoryPredictor,
)


def _center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return (float((x1 + x2) / 2.0), float((y1 + y2) / 2.0))


def predict_video_trajectory(
    video_path,
    output_csv,
    confidence=0.30,
    iou=0.65,
    target_classes=(4,),
    horizon_seconds=0.5,
    tracker="kalman",
):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    model = YOLO(MODEL_PATH)
    if tracker == "kalman":
        predictor = KalmanTrajectoryPredictor(horizon_seconds=horizon_seconds)
    else:
        predictor = TrajectoryPredictor(horizon_seconds=horizon_seconds)

    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_idx / fps
        results = model.predict(frame, conf=confidence, iou=iou, save=False, show=False, verbose=False)

        best = None
        best_conf = -1.0
        for result in results or []:
            if result.boxes is None or result.boxes.xyxy is None:
                continue
            for i in range(len(result.boxes.xyxy)):
                cls_id = int(result.boxes.cls[i]) if result.boxes.cls is not None else -1
                conf = float(result.boxes.conf[i]) if result.boxes.conf is not None else 0.0
                if cls_id in target_classes and conf > best_conf:
                    best_conf = conf
                    best = result.boxes.xyxy[i].tolist()

        if best is not None:
            cx, cy = _center_xyxy(best)
            state = predictor.update(cx, cy, timestamp)
            if state is not None:
                row = {
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 4),
                    "x": round(float(state.position[0]), 3),
                    "y": round(float(state.position[1]), 3),
                    "vx": round(float(state.velocity[0]), 4),
                    "vy": round(float(state.velocity[1]), 4),
                    "speed": round(float(state.speed), 4),
                    "heading_deg": round(float(state.heading_deg), 3),
                    "pred_x": round(float(state.predicted_position[0]), 3),
                    "pred_y": round(float(state.predicted_position[1]), 3),
                    "det_conf": round(float(best_conf), 4),
                }

                # Kalman-specific uncertainty fields.
                if hasattr(state, "heading_confidence"):
                    row["heading_confidence"] = round(float(state.heading_confidence), 4)
                    row["speed_confidence"] = round(float(state.speed_confidence), 4)
                    cov = getattr(state, "covariance")
                    row["pos_cov_trace"] = round(float(cov[0, 0] + cov[1, 1]), 4)
                    row["vel_cov_trace"] = round(float(cov[2, 2] + cov[3, 3]), 4)
                else:
                    row["heading_confidence"] = ""
                    row["speed_confidence"] = ""
                    row["pos_cov_trace"] = ""
                    row["vel_cov_trace"] = ""

                rows.append(row)

        frame_idx += 1

    cap.release()

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            f.write("frame,timestamp,x,y,vx,vy,speed,heading_deg,pred_x,pred_y,det_conf,heading_confidence,speed_confidence,pos_cov_trace,vel_cov_trace\n")

    return {
        "rows": len(rows),
        "output_csv": output_csv,
        "frames_processed": frame_idx,
        "tracker": tracker,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict short-horizon trajectory on a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-csv", default=f"trajectory_{int(time.time())}.csv")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--target-classes", default="4", help="Comma-separated class IDs to track.")
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument(
        "--tracker",
        choices=["simple", "kalman"],
        default="kalman",
        help="Tracking backend to use for trajectory state estimation.",
    )
    args = parser.parse_args()

    target_classes = tuple(int(x.strip()) for x in args.target_classes.split(",") if x.strip())
    summary = predict_video_trajectory(
        video_path=args.video,
        output_csv=args.output_csv,
        confidence=args.confidence,
        iou=args.iou,
        target_classes=target_classes,
        horizon_seconds=args.horizon_seconds,
        tracker=args.tracker,
    )
    print(summary)


if __name__ == "__main__":
    main()
