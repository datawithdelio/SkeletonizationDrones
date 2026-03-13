# SkeletonizationDrones Presentation Brief

Use this as the source for the Methods and Materials section, the Results section, and the References slide.

## Project Summary

This project is a web application for object-focused skeletonization. A user uploads an image or video in the frontend, the Flask backend runs YOLO-based segmentation to isolate the target object, converts the mask into contour geometry, generates a skeleton overlay, and returns the result for viewing or download.

For the current evaluation artifacts in this repo, the strongest measured evidence is on a `cars` vs `noncars` benchmark with `day` and `night` splits. The repo also contains smaller early evaluation artifacts for `drone/bird`, but the reliable presentation-ready KPIs come from the larger `cars_noncars_daynight` experiments.

## Methods And Materials

### Objective

Demonstrate an end-to-end pipeline that:

- detects relevant objects using a lightweight segmentation model,
- extracts a foreground mask,
- converts the mask into contours,
- generates a skeletonized representation,
- serves the result through a Flask + React application,
- evaluates performance with precision, recall, F1, accuracy, and inference latency.

### Software Stack

- Backend: Flask API
- Frontend: React + Vite
- Vision model: `yolov8n-seg.onnx`
- Core image processing: OpenCV
- Skeleton method benchmarking: Kimia EDF, OpenCV thinning, `skimage.skeletonize`, `skimage.medial_axis`

### Data Pipeline

The repo contains two dataset-preparation workflows:

- `prepare_kaggle_dataset.py`
  - converts a Kaggle-style labeled CSV or XLSX dataset into a folder structure grouped by `day/night` and binary classes,
  - supports configurable positive and negative labels,
  - supports copy or move behavior,
  - records skipped missing or unmapped rows.
- `prepare_media_eval_dataset.py`
  - builds an evaluation dataset from mixed image and video media,
  - classifies files by keywords in file paths,
  - extracts frames from videos,
  - organizes outputs into `day/night` and positive/negative folders.

### Dataset Used For Main Results

Main evaluation dataset:

- `backend/datasets/cars_noncars_daynight`
- Day cars: 700
- Day noncars: 700
- Night cars: 300
- Night noncars: 300
- Total samples: 2000

Threshold-tuning dataset:

- `backend/datasets/cars_noncars_daynight_tune`
- Day cars: 200
- Day noncars: 200
- Night cars: 100
- Night noncars: 100
- Total samples: 600

### Inference / Processing Pipeline

1. The frontend sends an image, video, or prompt request to the Flask backend.
2. The backend uses YOLO segmentation to detect target object instances.
3. The predicted mask is cleaned with basic morphology.
4. The binary mask is converted into contour strings.
5. The contour geometry is passed into the skeleton generator.
6. The result is overlaid on the original image or video frame.
7. The API returns the generated file and optional caption metadata.

### Tuned Parameters

The evaluation sweep varied:

- confidence threshold: `0.10`, `0.20`, `0.30`
- IoU threshold: `0.50`, `0.65`

### Metrics Reported

- Precision
- Recall
- F1 score
- Accuracy
- Mean latency in milliseconds
- 95th percentile latency
- Confusion matrix counts: TP, TN, FP, FN

### Important Presentation Note

Do not present a classic "feature importance" chart. This app is not a trained tabular classifier with interpretable input features. A stronger and more accurate substitute is:

- hyperparameter sensitivity,
- threshold sweep comparison,
- precision/recall tradeoff,
- day vs night robustness comparison.

## Results

### Main KPI Table: Baseline vs Tuned

The best final configuration in this repo is:

- confidence = `0.10`
- IoU = `0.50`

Compared against the baseline:

| Run | Confidence | IoU | Precision | Recall | F1 | Accuracy | Mean Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.30 | 0.65 | 0.9951 | 0.4070 | 0.5777 | 0.7025 | 138.064 | 167.428 |
| Tuned | 0.10 | 0.50 | 0.9863 | 0.5040 | 0.6671 | 0.7485 | 300.275 | 426.655 |

### KPI Improvement Summary

- Recall improved by `+9.70` percentage points
- F1 improved by `+8.94` percentage points
- Accuracy improved by `+4.60` percentage points
- Precision dropped by only `0.88` percentage points
- Mean latency increased by about `2.18x`

This gives a strong presentation story:

- the tuned system catches more true positives,
- overall balance improved,
- the tradeoff is slower inference.

### Day vs Night Performance For Best Final Run

| Split | Samples | Precision | Recall | F1 | Accuracy | Mean Latency (ms) | P95 Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Day | 1400 | 0.9860 | 0.5043 | 0.6673 | 0.7486 | 307.582 | 445.462 |
| Night | 600 | 0.9869 | 0.5033 | 0.6667 | 0.7483 | 283.226 | 372.278 |

### Interpretation

- Performance is very consistent across day and night.
- Night enhancement did not collapse performance.
- Specificity is very high in both splits.
- The system still misses many positives, so recall remains the main weakness.

### Aggregate Confusion Matrix For Best Final Run

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| 504 | 993 | 7 | 496 |

### Threshold Sweep Results

Use this table to show parameter sensitivity:

| Confidence | IoU | Precision | Recall | F1 | Accuracy | Mean Latency (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.50 | 0.9931 | 0.4767 | 0.6441 | 0.7367 | 266.568 |
| 0.10 | 0.65 | 0.9931 | 0.4767 | 0.6441 | 0.7367 | 275.083 |
| 0.20 | 0.65 | 0.9921 | 0.4200 | 0.5902 | 0.7083 | 289.413 |
| 0.20 | 0.50 | 0.9921 | 0.4200 | 0.5902 | 0.7083 | 336.645 |
| 0.30 | 0.50 | 0.9918 | 0.4033 | 0.5735 | 0.7000 | 301.522 |
| 0.30 | 0.65 | 0.9918 | 0.4033 | 0.5735 | 0.7000 | 342.265 |

### Threshold Sweep Takeaways

- Lowering confidence from `0.30` to `0.10` mattered more than changing IoU.
- In this sweep, IoU had almost no effect on precision, recall, F1, or accuracy.
- Confidence threshold was the main driver of recall and F1.
- `0.10 / 0.50` was selected because it matched the best metrics while keeping latency lower than `0.10 / 0.65`.

### Skeleton Method Benchmark

This comparison is useful for a methods/results graphic:

| Method | Runtime (ms) | Skeleton Pixels | Connected Components |
|---|---:|---:|---:|
| Kimia EDF | 1102.080 | 29095 | 60 |
| OpenCV Thinning | 520.528 | 2118 | 1 |
| skimage.skeletonize | 168.230 | 2152 | 1 |
| skimage.medial_axis | 145.999 | 3335 | 1 |

### Benchmark Interpretation

- Kimia EDF is much slower but produces a denser skeleton output.
- The classical skeleton methods are much faster.
- `skimage.medial_axis` was the fastest measured method in this benchmark set.
- A good presentation angle is that the app prioritizes richer geometric skeleton output over raw speed.

## What Graphics, Tables, And Charts To Include

### Best Graphics For Methods And Materials

- A pipeline diagram:
  `Upload -> YOLO segmentation -> mask cleaning -> contour extraction -> skeleton generation -> overlay -> download`
- A dataset composition bar chart:
  `day cars`, `day noncars`, `night cars`, `night noncars`
- A system architecture graphic:
  `React frontend`, `Flask backend`, `YOLO model`, `skeleton generator`, `output storage`

### Best Graphics For Results

- Baseline vs tuned grouped bar chart for:
  precision, recall, F1, accuracy
- Latency comparison chart:
  mean latency and p95 latency for baseline vs tuned
- Day vs night comparison chart:
  F1 or recall for day and night on the tuned run
- Threshold sweep line chart:
  x-axis = confidence threshold, y-axis = F1 and recall
- Confusion matrix table for the final tuned model
- Qualitative image panel:
  foreground mask + final skeleton outputs

### Existing Repo Assets You Can Use Immediately

- `backend/evaluation_outputs_cars/method_benchmark/foreground_mask.png`
- `backend/evaluation_outputs_cars/method_benchmark/kimia_edf.png`
- `backend/evaluation_outputs_cars/method_benchmark/opencv_thinning.png`
- `backend/evaluation_outputs_cars/method_benchmark/skimage_skeletonize.png`
- `backend/evaluation_outputs_cars/method_benchmark/skimage_medial_axis.png`
- `airplane_test.png`

## Suggested Speaker Notes

### Methods And Materials

"We built a Flask and React application that takes an uploaded image or video, segments the target object with a lightweight YOLO segmentation model, converts the object mask into contours, and generates a skeletonized representation. To make evaluation reproducible, we created scripts that reorganize Kaggle-style labeled data into day/night and binary-class folders, then ran a strict evaluation measuring precision, recall, F1, accuracy, and latency."

### Results

"Our baseline configuration achieved very high precision but lower recall, meaning the system was conservative and missed many positive cases. After threshold tuning, recall improved from 40.7% to 50.4%, F1 increased from 57.8% to 66.7%, and accuracy rose from 70.3% to 74.9%. The tradeoff was higher latency, which is expected when lowering the confidence threshold to detect more true positives."

## Limitations To State Clearly

- The repo proves strong evaluation on the `cars` vs `noncars` benchmark, not a full production-grade drone benchmark.
- The original Kaggle dataset title or URL is not explicitly preserved in the repo.
- Precision is excellent, but recall is still moderate.
- Faster skeletonization methods exist, but the chosen pipeline emphasizes richer geometry.

## References To Cite

### Local Project References

- `backend/skeleton_generation/experiments/prepare_kaggle_dataset.py`
- `backend/skeleton_generation/experiments/prepare_media_eval_dataset.py`
- `backend/skeleton_generation/experiments/evaluate_drone_vs_bird.py`
- `backend/skeleton_generation/experiments/run_full_evaluation.py`
- `backend/skeleton_generation/experiments/threshold_sweep_eval.py`
- `backend/skeleton_generation/experiments/benchmark_skeleton_methods.py`
- `backend/skeleton_generation/skel.py`
- `backend/evaluation_outputs_cars_real/strict_evaluation_report.json`
- `backend/evaluation_outputs_cars_real_tuned/strict_evaluation_report.json`
- `backend/evaluation_outputs_cars_real/threshold_sweep_tune/threshold_sweep_results.md`
- `IMPROVEMENTS_SUMMARY.txt`

### External References To Put On Slides

- Ultralytics documentation: https://docs.ultralytics.com/
- OpenCV morphology tutorial: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- scikit-image skeletonization example: https://scikit-image.org/docs/stable/auto_examples/edges/plot_skeleton.html
- scikit-image morphology API: https://scikit-image.org/docs/stable/api/skimage.morphology.html
- Flask quickstart: https://flask.palletsprojects.com/en/stable/quickstart/
- React quick start: https://react.dev/learn
- Vite getting started: https://vite.dev/guide/

### If You Have The Exact Dataset Source

Add the exact Kaggle dataset title and URL to the references slide. If you do not have the original link, do not invent one. In that case, say:

"Images were reorganized from a Kaggle-style labeled dataset and evaluated locally using the project preprocessing scripts."
