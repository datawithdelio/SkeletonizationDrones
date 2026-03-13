# SkeletonizationDrones Research Day Poster Guide

This file is the poster version of the project write-up. Use it for a tri-fold or wide academic poster like the example image you shared.

## Best Poster Format

Use a 3-column poster with:

- a large title bar across the top,
- authors and school under the title,
- left column for background and objective,
- center column for methods and visuals,
- right column for results, conclusion, and future work.

Keep each section short. The poster should be image-heavy and metric-heavy, not paragraph-heavy.

## Poster Title

Use one of these:

1. `SkeletonizationDrones: A YOLO-Based Object Skeletonization Web Application`
2. `Object Skeletonization Using Segmentation, Contour Extraction, and Geometric Processing`
3. `SkeletonizationDrones: An Interactive Pipeline for Object Detection and Skeleton Generation`

Recommended title:

`SkeletonizationDrones: A YOLO-Based Object Skeletonization Web Application`

## Author Line

Use:

`Your Name, Kumar, and any additional team members`

Then add:

`Department / Program Name`

`Research Days 2026`

## Poster Layout

### Top Banner

Put:

- title,
- author names,
- school name,
- one-sentence project summary.

Example summary:

`A Flask + React application that detects objects, extracts masks, and generates skeletonized visual outputs from uploaded images and videos.`

---

## Left Column

### 1. Background

Use this text:

`Object skeletonization reduces a detected shape to its structural centerline, making it easier to analyze geometry, shape, and spatial structure. This is useful in computer vision pipelines that need interpretable shape representations instead of only bounding boxes or segmentation masks.`

### 2. Problem Statement

Use this text:

`Although object detection and segmentation are common, many systems stop at mask generation and do not convert the object into a simplified skeletal representation. Our goal was to build an end-to-end application that accepts media input, isolates the object, generates a geometric skeleton, and returns a usable visual output through a web interface.`

### 3. Objective

Use this text:

`The objective of this project was to design a web-based pipeline for object skeletonization and evaluate how threshold tuning affects detection quality, robustness, and inference speed.`

### 4. Materials / Tools

Use these bullets:

- React + Vite frontend
- Flask backend API
- Ultralytics YOLO segmentation model (`yolov8n-seg.onnx`)
- OpenCV image processing
- scikit-image skeleton methods
- Custom geometric skeleton extraction pipeline

### 5. Dataset Summary

Use this table:

| Dataset Split | Positive | Negative | Total |
|---|---:|---:|---:|
| Day | 700 cars | 700 noncars | 1400 |
| Night | 300 cars | 300 noncars | 600 |
| Total | 1000 cars | 1000 noncars | 2000 |

Use this note under it:

`A second tuning dataset of 600 samples was used for confidence/IoU threshold selection.`

### Visuals For Left Column

Use:

- [demo-drone.png](/Users/deliorincon/SkeletonizationDrones/frontend/public/demo-drone.png)
- [airplane_test.png](/Users/deliorincon/SkeletonizationDrones/airplane_test.png)

If you only use one image here, use `airplane_test.png`.

---

## Center Column

### 6. Methodology

Use this short workflow:

1. User uploads image or video.
2. Flask backend receives the file.
3. YOLO segmentation detects object regions.
4. The binary mask is cleaned with morphology.
5. Contours are extracted from the mask.
6. Skeleton generation is applied to the contour geometry.
7. The final skeleton is overlaid on the original media.

### 7. System Pipeline Graphic

Make a simple horizontal diagram:

`Input Media -> YOLO Segmentation -> Mask Cleanup -> Contour Extraction -> Skeleton Generation -> Overlay Output`

### 8. Skeleton Method Comparison

Use this table:

| Method | Runtime (ms) | Skeleton Pixels | Connected Components |
|---|---:|---:|---:|
| Kimia EDF | 1102.080 | 29095 | 60 |
| OpenCV Thinning | 520.528 | 2118 | 1 |
| skimage.skeletonize | 168.230 | 2152 | 1 |
| skimage.medial_axis | 145.999 | 3335 | 1 |

### 9. Methodology Interpretation

Use this paragraph:

`The system combines learned segmentation with classical geometric processing. YOLO is used to locate object masks, while contour extraction and skeletonization convert the mask into a compact structural representation. This hybrid approach preserves interpretability while still using a modern detection backbone.`

### Visuals For Center Column

Best sequence:

- [foreground_mask.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/foreground_mask.png)
- [kimia_edf.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/kimia_edf.png)
- [opencv_thinning.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/opencv_thinning.png)
- [skimage_skeletonize.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/skimage_skeletonize.png)
- [skimage_medial_axis.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/skimage_medial_axis.png)

Best center-column layout:

- one row with `foreground_mask.png`,
- one row with 3 skeleton outputs,
- or one large `kimia_edf.png` if space is limited.

---

## Right Column

### 10. Results

Use this baseline-vs-tuned KPI table:

| Run | Confidence | IoU | Precision | Recall | F1 | Accuracy | Mean Latency (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.30 | 0.65 | 0.9951 | 0.4070 | 0.5777 | 0.7025 | 138.064 |
| Tuned | 0.10 | 0.50 | 0.9863 | 0.5040 | 0.6671 | 0.7485 | 300.275 |

### 11. Threshold Sweep

Use this table:

| Confidence | IoU | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.50 | 0.9931 | 0.4767 | 0.6441 | 0.7367 |
| 0.10 | 0.65 | 0.9931 | 0.4767 | 0.6441 | 0.7367 |
| 0.20 | 0.65 | 0.9921 | 0.4200 | 0.5902 | 0.7083 |
| 0.20 | 0.50 | 0.9921 | 0.4200 | 0.5902 | 0.7083 |
| 0.30 | 0.50 | 0.9918 | 0.4033 | 0.5735 | 0.7000 |
| 0.30 | 0.65 | 0.9918 | 0.4033 | 0.5735 | 0.7000 |

### 12. Key Findings

Use these bullets:

- Lower confidence thresholds improved recall and F1.
- IoU threshold had minimal impact compared with confidence tuning.
- The final tuned configuration improved balanced performance.
- Precision remained very high across all experiments.
- The main tradeoff was higher latency.

### 13. Day vs Night Robustness

Use this table:

| Split | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Day | 0.9860 | 0.5043 | 0.6673 | 0.7486 |
| Night | 0.9869 | 0.5033 | 0.6667 | 0.7483 |

Use this line under the table:

`Performance remained highly consistent across day and night subsets, indicating stable behavior under different lighting conditions.`

### 14. Conclusion

Use this paragraph:

`SkeletonizationDrones successfully demonstrates an end-to-end pipeline for object segmentation and skeleton generation within a usable web application. Threshold tuning improved recall, F1, and overall accuracy while maintaining very high precision. The results support the feasibility of combining lightweight segmentation with geometric skeleton extraction for interpretable visual analysis.`

### 15. Future Work

Use these bullets:

- Evaluate on a larger drone-specific dataset
- Improve recall without doubling latency
- Add stronger benchmarking on videos
- Compare more skeletonization strategies
- Deploy a polished public demo

### 16. References

Keep references short on the poster:

- Ultralytics YOLO Documentation
- OpenCV Documentation
- scikit-image Morphology Documentation
- Flask Documentation
- React and Vite Documentation

If space allows, also cite:

- [prepare_kaggle_dataset.py](/Users/deliorincon/SkeletonizationDrones/backend/skeleton_generation/experiments/prepare_kaggle_dataset.py)
- [run_full_evaluation.py](/Users/deliorincon/SkeletonizationDrones/backend/skeleton_generation/experiments/run_full_evaluation.py)
- [threshold_sweep_eval.py](/Users/deliorincon/SkeletonizationDrones/backend/skeleton_generation/experiments/threshold_sweep_eval.py)

---

## Best Visual Set For The Poster

If you only use 5 visuals total, use these:

1. [airplane_test.png](/Users/deliorincon/SkeletonizationDrones/airplane_test.png)
2. [foreground_mask.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/foreground_mask.png)
3. [kimia_edf.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/kimia_edf.png)
4. [skimage_skeletonize.png](/Users/deliorincon/SkeletonizationDrones/backend/evaluation_outputs_cars/method_benchmark/skimage_skeletonize.png)
5. [airplane_test-skeleton.png](/Users/deliorincon/SkeletonizationDrones/backend/output_path/airplane_test-skeleton.png)

## What To Avoid

Do not put these on the poster:

- raw JSON screenshots,
- full terminal output,
- giant paragraphs,
- claims about "feature importance",
- claims that the current evaluation is a full drone benchmark if you only show the car/noncar metrics.

## Fast Build Strategy

If you are making this fast in PowerPoint, Canva, or Google Slides:

1. Create a wide poster with 3 columns.
2. Add the title and author bar first.
3. Put Background, Problem, Objective, and Dataset in the left column.
4. Put Methodology and all image examples in the center column.
5. Put KPI tables, findings, conclusion, and future work in the right column.
6. Keep each text box short and use bold headings.
7. Export as PDF for printing.

## Best One-Sentence Pitch

Use this when people walk up to the poster:

`We built a web app that takes an uploaded image or video, segments the main object, and converts it into a geometric skeleton so the shape becomes easier to analyze and visualize.`
