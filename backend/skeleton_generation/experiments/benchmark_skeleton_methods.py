import argparse
import json
import os
import time
from dataclasses import dataclass

import cv2 as cv
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    os.path.dirname(CURRENT_DIR), "utils", "models", "yolov8n-seg.onnx"
)


@dataclass
class MethodResult:
    name: str
    runtime_ms: float
    skeleton_pixels: int
    connected_components: int
    output_path: str
    error: str = ""


def _safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def _load_image(path):
    image = cv.imread(path, cv.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _extract_foreground_mask_with_yolo(image, confidence=0.35, iou=0.65):
    model = YOLO(MODEL_PATH)
    results = model.predict(image, conf=confidence, iou=iou, save=False, show=False, verbose=False)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    total_instances = 0
    for result in results or []:
        if result is None or result.masks is None:
            continue
        for instance in result:
            if instance.masks is None or not instance.masks.xy:
                continue
            total_instances += 1
            for poly in instance.masks.xy:
                contour = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv.drawContours(mask, [contour], -1, 255, cv.FILLED)

    return mask, total_instances


def _fallback_mask(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    border = np.concatenate((binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]))
    if np.mean(border) > 127:
        binary = cv.bitwise_not(binary)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)
    binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _count_components(binary):
    n_labels, _ = cv.connectedComponents((binary > 0).astype(np.uint8))
    return max(0, n_labels - 1)


def _save_binary(binary, path):
    cv.imwrite(path, (binary > 0).astype(np.uint8) * 255)


def _to_contour_strings(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    height = mask.shape[0]
    contour_strings = []
    for contour in contours:
        for point in contour:
            contour_strings.append("{:.7e} {:.7e}".format(float(point[0][0]), float(height - point[0][1])))
    return contour_strings


def run_kimia(mask, width, height):
    # Lazy import so other methods can run without full geometric dependencies.
    from skeleton_generation.utils.skeleton.extractKimiaEDF import generate_skeleton

    contour_strings = _to_contour_strings(mask)
    if not contour_strings:
        return np.zeros((height, width), dtype=np.uint8)
    skeleton_bgr = generate_skeleton(
        contour_strings,
        width,
        height,
        smooth_sigma=7,
        down_factor=1,
        points_data=[],
    )
    gray = cv.cvtColor(skeleton_bgr, cv.COLOR_BGR2GRAY)
    # Kimia renderer uses a near-white background; extract darker skeleton strokes.
    _, binary = cv.threshold(gray, 245, 255, cv.THRESH_BINARY_INV)
    return binary


def run_thinning(mask):
    if not hasattr(cv, "ximgproc") or not hasattr(cv.ximgproc, "thinning"):
        try:
            from skimage.morphology import skeletonize
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("opencv ximgproc and scikit-image are unavailable") from exc
        skel = skeletonize(mask > 0).astype(np.uint8) * 255
        return skel
    return cv.ximgproc.thinning(mask)


def run_skimage_skeletonize(mask):
    from skimage.morphology import skeletonize
    return (skeletonize(mask > 0).astype(np.uint8) * 255)


def run_skimage_medial(mask):
    from skimage.morphology import medial_axis
    med, _ = medial_axis(mask > 0, return_distance=True)
    return med.astype(np.uint8) * 255


def benchmark(image_path, output_dir, confidence, iou):
    _safe_mkdir(output_dir)
    image = _load_image(image_path)
    height, width = image.shape[:2]

    yolo_mask, yolo_instances = _extract_foreground_mask_with_yolo(image, confidence=confidence, iou=iou)
    if cv.countNonZero(yolo_mask) == 0:
        mask = _fallback_mask(image)
        mask_source = "fallback_otsu"
    else:
        mask = yolo_mask
        mask_source = "yolo_union_mask"

    _save_binary(mask, os.path.join(output_dir, "foreground_mask.png"))

    methods = {
        "kimia_edf": lambda m: run_kimia(m, width, height),
        "opencv_thinning": run_thinning,
        "skimage_skeletonize": run_skimage_skeletonize,
        "skimage_medial_axis": run_skimage_medial,
    }

    method_results = []
    for name, fn in methods.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        t0 = time.perf_counter()
        try:
            skeleton = fn(mask)
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            _save_binary(skeleton, output_path)
            method_results.append(
                MethodResult(
                    name=name,
                    runtime_ms=runtime_ms,
                    skeleton_pixels=int(np.count_nonzero(skeleton)),
                    connected_components=_count_components(skeleton),
                    output_path=output_path,
                )
            )
        except Exception as exc:
            runtime_ms = (time.perf_counter() - t0) * 1000.0
            method_results.append(
                MethodResult(
                    name=name,
                    runtime_ms=runtime_ms,
                    skeleton_pixels=0,
                    connected_components=0,
                    output_path=output_path,
                    error=str(exc),
                )
            )

    report = {
        "image_path": image_path,
        "mask_source": mask_source,
        "yolo_instances_detected": int(yolo_instances),
        "mask_pixels": int(np.count_nonzero(mask)),
        "methods": [vars(m) for m in method_results],
    }

    report_path = os.path.join(output_dir, "benchmark_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple skeletonization methods.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--output-dir", default="benchmark_outputs", help="Directory for outputs.")
    parser.add_argument("--confidence", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.65, help="YOLO IoU threshold.")
    args = parser.parse_args()

    report_path = benchmark(
        image_path=args.image,
        output_dir=args.output_dir,
        confidence=args.confidence,
        iou=args.iou,
    )
    print(f"Benchmark complete. Report: {report_path}")


if __name__ == "__main__":
    main()
