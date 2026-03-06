import argparse
import json
import time
from collections import deque

import cv2 as cv
from ultralytics import YOLO

from skeleton_generation.experiments.check_yolo_boxes import MODEL_PATH
from skeleton_generation.experiments.predict_trajectory_video import _center_xyxy
from skeleton_generation.utils.tracking.trajectory import KalmanTrajectoryPredictor


def _open_capture(source: str):
    if source == "webcam":
        return cv.VideoCapture(0)
    return cv.VideoCapture(source)


def run_live_telemetry(
    source: str,
    output_jsonl: str,
    confidence: float,
    iou: float,
    target_classes,
    horizon_seconds: float,
    summary_interval_frames: int,
    max_frames: int,
):
    cap = _open_capture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")

    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    model = YOLO(MODEL_PATH)
    tracker = KalmanTrajectoryPredictor(horizon_seconds=horizon_seconds)

    recent_conf = deque(maxlen=summary_interval_frames)
    recent_speed = deque(maxlen=summary_interval_frames)
    recent_heading = deque(maxlen=summary_interval_frames)

    frame_idx = 0
    summaries = 0

    with open(output_jsonl, "w", encoding="utf-8") as out:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames > 0 and frame_idx >= max_frames:
                break

            timestamp = frame_idx / fps
            t0 = time.perf_counter()
            results = model.predict(frame, conf=confidence, iou=iou, save=False, show=False, verbose=False)
            infer_latency_ms = (time.perf_counter() - t0) * 1000.0

            best = None
            best_conf = -1.0
            for result in results or []:
                boxes = getattr(result, "boxes", None)
                if boxes is None or boxes.xyxy is None:
                    continue
                for i in range(len(boxes.xyxy)):
                    cls_id = int(boxes.cls[i]) if boxes.cls is not None else -1
                    conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                    if cls_id in target_classes and conf > best_conf:
                        best_conf = conf
                        best = boxes.xyxy[i].tolist()

            state = None
            if best is not None:
                cx, cy = _center_xyxy(best)
                state = tracker.update(cx, cy, timestamp)
                recent_conf.append(max(0.0, best_conf))
                if state is not None:
                    recent_speed.append(float(state.speed))
                    recent_heading.append(float(state.heading_deg))

            if frame_idx > 0 and frame_idx % summary_interval_frames == 0:
                payload = {
                    "frame": frame_idx,
                    "timestamp": round(timestamp, 3),
                    "source": source,
                    "summary_window_frames": summary_interval_frames,
                    "mean_detection_conf": round(sum(recent_conf) / max(1, len(recent_conf)), 4),
                    "mean_speed": round(sum(recent_speed) / max(1, len(recent_speed)), 4) if recent_speed else 0.0,
                    "mean_heading_deg": round(sum(recent_heading) / max(1, len(recent_heading)), 3) if recent_heading else 0.0,
                    "infer_latency_ms": round(infer_latency_ms, 3),
                    "prediction": None,
                }

                if state is not None:
                    payload["prediction"] = {
                        "x": round(float(state.predicted_position[0]), 3),
                        "y": round(float(state.predicted_position[1]), 3),
                        "heading_confidence": round(float(state.heading_confidence), 4),
                        "speed_confidence": round(float(state.speed_confidence), 4),
                    }

                out.write(json.dumps(payload) + "\n")
                summaries += 1
                print(payload)

            frame_idx += 1

    cap.release()
    return {
        "output_jsonl": output_jsonl,
        "frames_processed": frame_idx,
        "summaries_written": summaries,
    }


def main():
    parser = argparse.ArgumentParser(description="Live mode: YOLO detection + Kalman telemetry summaries.")
    parser.add_argument("--source", default="webcam", help="`webcam` or video path")
    parser.add_argument("--output-jsonl", default="live_telemetry.jsonl")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.65)
    parser.add_argument("--target-classes", default="4", help="Comma-separated class IDs")
    parser.add_argument("--horizon-seconds", type=float, default=0.5)
    parser.add_argument("--summary-interval-frames", type=int, default=10)
    parser.add_argument("--max-frames", type=int, default=0, help="0 for unlimited")
    args = parser.parse_args()

    target_classes = tuple(int(x.strip()) for x in args.target_classes.split(",") if x.strip())
    summary = run_live_telemetry(
        source=args.source,
        output_jsonl=args.output_jsonl,
        confidence=args.confidence,
        iou=args.iou,
        target_classes=target_classes,
        horizon_seconds=args.horizon_seconds,
        summary_interval_frames=max(1, args.summary_interval_frames),
        max_frames=max(0, args.max_frames),
    )
    print(summary)


if __name__ == "__main__":
    main()
