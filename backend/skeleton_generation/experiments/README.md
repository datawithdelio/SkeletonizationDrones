# Skeletonization Experiments

This folder helps compare multiple skeletonization approaches and inspect detection quality.

## 1) Compare Skeletonization Methods (Speed + Output Stats)

```bash
cd backend
python -m skeleton_generation.experiments.benchmark_skeleton_methods \
  --image ../airplane_test.png \
  --output-dir ./benchmark_outputs \
  --confidence 0.35 \
  --iou 0.65
```

Outputs:
- `foreground_mask.png`
- `kimia_edf.png`
- `opencv_thinning.png`
- `skimage_skeletonize.png`
- `skimage_medial_axis.png`
- `benchmark_report.json` (runtime ms, skeleton pixels, connected components)

## 2) Inspect YOLO Boxes

```bash
cd backend
python -m skeleton_generation.experiments.check_yolo_boxes \
  --image ../airplane_test.png \
  --confidence 0.25 \
  --iou 0.65
```

Prints detection boxes, confidences, and class IDs as JSON.

## 3) Trajectory Prediction Utility

Use `skeleton_generation/utils/tracking/trajectory.py` to estimate heading angle and short-term predicted position from tracked 2D points.

## 4) Evaluate Drone-vs-Bird (Reviewer-Focused)

Dataset layout:

```text
your_dataset/
  drones/
  birds/
```

Run:

```bash
cd backend
python -m skeleton_generation.experiments.evaluate_drone_vs_bird \
  --dataset-dir /path/to/your_dataset \
  --confidence 0.30 \
  --iou 0.65 \
  --target-classes 4 \
  --night-mode \
  --output drone_vs_bird_report.json
```

Notes:
- `target-classes 4` maps to YOLO airplane-like class in this model setup.
- Use `--night-mode` for low-light enhancement before inference.

## 5) Predict Drone Trajectory From Video

```bash
cd backend
python -m skeleton_generation.experiments.predict_trajectory_video \
  --video /path/to/video.mp4 \
  --output-csv trajectory.csv \
  --confidence 0.30 \
  --iou 0.65 \
  --target-classes 4 \
  --horizon-seconds 0.5
```

Outputs per tracked frame:
- position `(x, y)`
- velocity `(vx, vy)`
- speed
- heading angle in degrees
- predicted future position `(pred_x, pred_y)`
- detection confidence
- (kalman mode) heading/speed confidence + covariance traces

## 6) Strict Evaluation Pipeline (Day/Night + Confusion + Latency)

Dataset layout (preferred):

```text
your_dataset/
  day/
    drones/
    birds/
  night/
    drones/
    birds/
```

Run:

```bash
cd backend
python -m skeleton_generation.experiments.run_full_evaluation \
  --dataset-dir /path/to/your_dataset \
  --output-dir ./evaluation_outputs \
  --confidence 0.30 \
  --iou 0.65 \
  --target-classes 4 \
  --benchmark-image ../airplane_test.png
```

Output:
- `strict_evaluation_report.json`
  - day/night metrics
  - aggregate confusion matrix
  - latency mean/p95
  - optional method comparison table

## 7) Multi-Image Method Comparison Table

```bash
cd backend
python -m skeleton_generation.experiments.build_method_comparison_table \
  --images ../airplane_test.png ../sample_image.png ../test_upload.png \
  --output-dir ./benchmark_outputs/multi \
  --confidence 0.35 \
  --iou 0.65
```

Outputs:
- `method_comparison_table.json`
- `method_comparison_table.md`

## 8) Live Mode + Compressed Telemetry Summary

Webcam:

```bash
cd backend
python -m skeleton_generation.experiments.live_stream_telemetry \
  --source webcam \
  --output-jsonl live_telemetry.jsonl \
  --summary-interval-frames 10 \
  --confidence 0.30 \
  --iou 0.65
```

Video file:

```bash
cd backend
python -m skeleton_generation.experiments.live_stream_telemetry \
  --source /path/to/video.mp4 \
  --output-jsonl live_telemetry.jsonl \
  --summary-interval-frames 10 \
  --max-frames 600
```

Each JSONL row is a compressed telemetry summary window with:
- average detection confidence
- average speed and heading
- infer latency
- predicted position + confidence

## 9) Prepare Kaggle Dataset (CSV/XLSX -> day/night drones/birds)

If your Kaggle dataset has labels in CSV/XLSX:

```bash
cd backend
python -m skeleton_generation.experiments.prepare_kaggle_dataset \
  --labels-file /path/to/labels.csv \
  --images-dir /path/to/images \
  --output-dir /path/to/converted_dataset \
  --filename-col filename \
  --label-col label \
  --daynight-col time_of_day \
  --drone-values drone,uav,quadcopter \
  --bird-values bird \
  --day-values day,daytime \
  --night-values night,nighttime,lowlight
```

If your labels do not have day/night column, keep everything as day by default:

```bash
cd backend
python -m skeleton_generation.experiments.prepare_kaggle_dataset \
  --labels-file /path/to/labels.csv \
  --images-dir /path/to/images \
  --output-dir /path/to/converted_dataset \
  --filename-col filename \
  --label-col label \
  --default-period day
```

You can test mapping without copying files:

```bash
cd backend
python -m skeleton_generation.experiments.prepare_kaggle_dataset \
  --labels-file /path/to/labels.csv \
  --images-dir /path/to/images \
  --output-dir /path/to/converted_dataset \
  --dry-run
```
