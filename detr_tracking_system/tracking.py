from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import uuid

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import torch


MODEL_NAME = "yolov8n.pt"
DEFAULT_TARGET_LABELS = ("airplane", "bird", "kite")
LABEL_ALIASES = {
    "drone": ("airplane", "bird", "kite"),
    "uav": ("airplane", "bird", "kite"),
    "quadcopter": ("airplane", "bird", "kite"),
}
TRACK_COLOR = (72, 220, 160)
PREDICTED_COLOR = (0, 170, 255)
TRAIL_COLOR = (46, 164, 255)
TEXT_COLOR = (255, 255, 255)
BOX_FILL_COLOR = (26, 70, 66)
FORECAST_COLOR = (120, 220, 255)
SKELETON_COLOR = (130, 255, 245)
SKELETON_GLOW = (255, 255, 255)


@dataclass(slots=True)
class Detection:
    bbox_xywh: list[float]
    confidence: float
    label: str


@dataclass(slots=True)
class TrackingConfig:
    confidence_threshold: float = 0.25
    target_labels: tuple[str, ...] = DEFAULT_TARGET_LABELS
    trail_length: int = 42
    resize_width: int = 512
    drone_mode: bool = False
    tile_stride: int = 6
    tile_threshold: float = 0.55
    max_age: int = 45
    n_init: int = 1
    nn_budget: int = 100
    max_predicted_frames: int = 18
    jpeg_quality: int = 70
    detect_interval: int = 2
    forecast_steps: int = 5
    motion_min_area: int = 700
    skeleton_refresh_interval: int = 3


def parse_target_labels(raw_value: str | None) -> tuple[str, ...]:
    if not raw_value:
        return DEFAULT_TARGET_LABELS

    normalized: list[str] = []
    for label in raw_value.split(","):
        clean = label.strip().lower()
        if not clean:
            continue
        alias_value = LABEL_ALIASES.get(clean, (clean,))
        normalized.extend(alias_value)
    # Preserve order while removing duplicates.
    return tuple(dict.fromkeys(normalized or DEFAULT_TARGET_LABELS))


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_tracking_config(form_data: dict[str, str]) -> TrackingConfig:
    def parse_int(name: str, default: int, minimum: int) -> int:
        value = form_data.get(name)
        parsed = int(value) if value not in (None, "") else default
        return max(minimum, parsed)

    def parse_float(name: str, default: float, minimum: float, maximum: float) -> float:
        value = form_data.get(name)
        parsed = float(value) if value not in (None, "") else default
        return min(maximum, max(minimum, parsed))

    return TrackingConfig(
        confidence_threshold=parse_float("confidence_threshold", 0.25, 0.01, 0.99),
        target_labels=parse_target_labels(form_data.get("target_labels")),
        trail_length=parse_int("trail_length", 42, 2),
        resize_width=parse_int("resize_width", 512, 0),
        drone_mode=parse_bool(form_data.get("drone_mode"), False),
        tile_stride=parse_int("tile_stride", 6, 1),
        tile_threshold=parse_float("tile_threshold", 0.55, 0.05, 0.99),
        max_age=parse_int("max_age", 45, 1),
        n_init=parse_int("n_init", 1, 1),
        nn_budget=parse_int("nn_budget", 100, 1),
        max_predicted_frames=parse_int("max_predicted_frames", 18, 0),
        jpeg_quality=parse_int("jpeg_quality", 70, 30),
        detect_interval=parse_int("detect_interval", 2, 1),
        forecast_steps=parse_int("forecast_steps", 5, 1),
        motion_min_area=parse_int("motion_min_area", 700, 100),
        skeleton_refresh_interval=parse_int("skeleton_refresh_interval", 3, 1),
    )


class YoloDroneDetector:
    def __init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics YOLO is not installed. Run `pip install ultralytics==8.2.75` in the virtualenv."
            ) from exc

        self.device = self._resolve_device()
        self.device_label = str(self.device)
        self.uses_cuda = self.device_label == "0"
        self.model = YOLO(MODEL_NAME)
        self.class_names = self.model.names
        self.name_to_id = {
            name.lower(): int(class_id)
            for class_id, name in self.class_names.items()
        }

    def detect(self, frame: np.ndarray, config: TrackingConfig, frame_index: int) -> list[Detection]:
        detections = self._detect_patch(
            frame=frame,
            config=config,
            threshold=config.confidence_threshold,
            offset_x=0,
            offset_y=0,
        )

        should_run_tile_pass = (
            config.drone_mode
            and frame_index % config.tile_stride == 0
            and min(frame.shape[:2]) >= 320
        )
        if should_run_tile_pass:
            detections.extend(self._detect_tiled(frame, config))

        return self._non_max_suppress(detections, iou_threshold=0.45)

    def _detect_tiled(self, frame: np.ndarray, config: TrackingConfig) -> list[Detection]:
        height, width = frame.shape[:2]
        overlap_ratio = 0.2
        tile_width = max(224, int(width * 0.6))
        tile_height = max(224, int(height * 0.6))
        step_x = max(1, int(tile_width * (1.0 - overlap_ratio)))
        step_y = max(1, int(tile_height * (1.0 - overlap_ratio)))

        positions_x = self._tile_positions(width, tile_width, step_x)
        positions_y = self._tile_positions(height, tile_height, step_y)
        detections: list[Detection] = []

        for top in positions_y:
            for left in positions_x:
                patch = frame[top:top + tile_height, left:left + tile_width]
                detections.extend(
                    self._detect_patch(
                        frame=patch,
                        config=config,
                        threshold=min(config.confidence_threshold, config.tile_threshold),
                        offset_x=left,
                        offset_y=top,
                    )
                )
        return detections

    def _tile_positions(self, full_size: int, tile_size: int, step: int) -> list[int]:
        if tile_size >= full_size:
            return [0]

        positions = list(range(0, max(1, full_size - tile_size + 1), step))
        final_position = full_size - tile_size
        if positions[-1] != final_position:
            positions.append(final_position)
        return positions

    def _detect_patch(
        self,
        frame: np.ndarray,
        config: TrackingConfig,
        threshold: float,
        offset_x: int,
        offset_y: int,
    ) -> list[Detection]:
        class_ids = self._target_class_ids(config.target_labels)
        results = self.model.predict(
            source=frame,
            conf=threshold,
            classes=class_ids or None,
            imgsz=self._prediction_size(frame),
            verbose=False,
            device=self.device_label,
        )

        detections: list[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            score = float(box.conf.item())
            label_index = int(box.cls.item())
            label_name = str(self.class_names[label_index]).lower()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1 += offset_x
            x2 += offset_x
            y1 += offset_y
            y2 += offset_y
            x1 = max(0, x1)
            y1 = max(0, y1)
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            if width < 2 or height < 2:
                continue

            detections.append(
                Detection(
                    bbox_xywh=[x1, y1, width, height],
                    confidence=score,
                    label=label_name,
                )
            )
        return detections

    def _resolve_device(self) -> str:
        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _target_class_ids(self, target_labels: tuple[str, ...]) -> list[int]:
        class_ids: list[int] = []
        for label in target_labels:
            class_id = self.name_to_id.get(label.lower())
            if class_id is not None:
                class_ids.append(class_id)
        return list(dict.fromkeys(class_ids))

    def _prediction_size(self, frame: np.ndarray) -> int:
        height, width = frame.shape[:2]
        longest_side = max(height, width)
        if longest_side <= 416:
            return 416
        return 512

    def _non_max_suppress(self, detections: list[Detection], iou_threshold: float) -> list[Detection]:
        ordered = sorted(detections, key=lambda det: det.confidence, reverse=True)
        kept: list[Detection] = []

        for candidate in ordered:
            if all(self._iou(candidate, existing) < iou_threshold for existing in kept):
                kept.append(candidate)
        return kept

    def _iou(self, first: Detection, second: Detection) -> float:
        ax1, ay1, aw, ah = first.bbox_xywh
        bx1, by1, bw, bh = second.bbox_xywh
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union_area = (aw * ah) + (bw * bh) - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


class DetectorFactory:
    def __init__(self) -> None:
        self._detector: YoloDroneDetector | None = None
        self._lock = threading.Lock()

    def get(self) -> YoloDroneDetector:
        with self._lock:
            if self._detector is None:
                self._detector = YoloDroneDetector()
            return self._detector


class VideoTrackingSession:
    def __init__(
        self,
        session_id: str,
        video_path: Path,
        detector: YoloDroneDetector,
        config: TrackingConfig,
    ) -> None:
        self.session_id = session_id
        self.video_path = video_path
        self.detector = detector
        self.config = config
        self.condition = threading.Condition()
        self.processing_thread: threading.Thread | None = None
        self.latest_frame: bytes | None = None
        self.latest_frame_index = -1
        self.completed = False
        self.error: str | None = None
        self.started = False
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.fps = 0.0
        self.source_fps = 0.0
        self.processed_frames = 0
        self.total_frames = 0
        self.active_tracks = 0
        self.predicted_tracks = 0
        self.seen_track_ids: set[int] = set()
        self.display_track_ids: dict[int, int] = {}
        self.next_display_track_id = 1
        self.trails: dict[int, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.config.trail_length)
        )
        self.skeleton_cache: dict[int, tuple[int, np.ndarray]] = {}
        self.previous_gray_frame: np.ndarray | None = None

    def start(self) -> None:
        with self.condition:
            if self.started:
                return
            self.started = True
            self.processing_thread = threading.Thread(
                target=self._process_video,
                name=f"tracking-session-{self.session_id}",
                daemon=True,
            )
            self.processing_thread.start()

    def get_status(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "video_path": str(self.video_path),
            "device": self.detector.device_label,
            "started": self.started,
            "completed": self.completed,
            "error": self.error,
            "source_fps": round(self.source_fps, 2),
            "processing_fps": round(self.fps, 2),
            "processed_frames": self.processed_frames,
            "total_frames": self.total_frames,
            "active_tracks": self.active_tracks,
            "predicted_tracks": self.predicted_tracks,
            "unique_tracks_seen": len(self.seen_track_ids),
            "target_labels": list(self.config.target_labels),
            "confidence_threshold": self.config.confidence_threshold,
            "drone_mode": self.config.drone_mode,
        }

    def mjpeg_stream(self):
        self.start()
        last_frame_index = -1

        while True:
            with self.condition:
                if self.latest_frame_index == last_frame_index and not self.completed and not self.error:
                    self.condition.wait(timeout=0.5)

                frame = self.latest_frame
                frame_index = self.latest_frame_index
                completed = self.completed
                error = self.error

            if frame is not None and frame_index != last_frame_index:
                last_frame_index = frame_index
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                continue

            if error:
                break

            if completed and frame_index == last_frame_index:
                break

    def _process_video(self) -> None:
        self.started_at = time.perf_counter()
        capture: cv2.VideoCapture | None = None

        try:
            tracker = DeepSort(
                max_age=self.config.max_age,
                n_init=self.config.n_init,
                nn_budget=self.config.nn_budget,
                embedder="mobilenet",
                embedder_gpu=self.detector.uses_cuda,
                half=self.detector.uses_cuda,
                bgr=True,
            )

            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                self._finish_with_error(f"Unable to open video: {self.video_path}")
                return

            self.source_fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
            self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            frame_delay = 1.0 / self.source_fps if self.source_fps > 0 else 0.0

            while True:
                frame_started = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    break

                frame = self._resize_frame(frame)
                frame_index = self.processed_frames + 1
                should_detect = (
                    frame_index == 1
                    or self.config.detect_interval <= 1
                    or frame_index % self.config.detect_interval == 0
                )
                detections = self.detector.detect(frame, self.config, frame_index) if should_detect else []
                if not detections:
                    detections = self._motion_fallback_detect(frame)
                track_inputs = [
                    (det.bbox_xywh, det.confidence, det.label)
                    for det in detections
                ]
                tracks = tracker.update_tracks(track_inputs, frame=frame)
                annotated = self._render_tracks(frame, tracks)
                encoded_ok, encoded_frame = cv2.imencode(
                    ".jpg",
                    annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )
                if not encoded_ok:
                    raise RuntimeError("OpenCV failed to JPEG-encode a frame.")

                self.processed_frames += 1
                elapsed_total = max(time.perf_counter() - self.started_at, 1e-6)
                self.fps = self.processed_frames / elapsed_total

                with self.condition:
                    self.latest_frame = encoded_frame.tobytes()
                    self.latest_frame_index = self.processed_frames
                    self.condition.notify_all()

                frame_elapsed = time.perf_counter() - frame_started
                if frame_delay > 0 and frame_elapsed < frame_delay:
                    time.sleep(frame_delay - frame_elapsed)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            self._finish_with_error(str(exc))
            if capture is not None:
                capture.release()
            return

        if capture is not None:
            capture.release()
        self.finished_at = time.perf_counter()
        with self.condition:
            self.completed = True
            self.condition.notify_all()

    def _finish_with_error(self, message: str) -> None:
        self.error = message
        self.finished_at = time.perf_counter()
        with self.condition:
            self.completed = True
            self.condition.notify_all()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.config.resize_width <= 0:
            return frame

        height, width = frame.shape[:2]
        if width <= self.config.resize_width:
            return frame

        ratio = self.config.resize_width / float(width)
        resized_height = max(1, int(height * ratio))
        return cv2.resize(frame, (self.config.resize_width, resized_height), interpolation=cv2.INTER_AREA)

    def _render_tracks(self, frame: np.ndarray, tracks) -> np.ndarray:
        visible_track_ids: set[int] = set()
        forecasted_track_ids: set[int] = set()
        display_tracks = self._select_display_tracks(frame, tracks)

        for track in display_tracks:
            track_id = int(track.track_id)
            is_predicted = track.time_since_update > 0

            visible_track_ids.add(track_id)
            self.seen_track_ids.add(track_id)
            display_track_id = self.display_track_ids.setdefault(track_id, self.next_display_track_id)
            if display_track_id == self.next_display_track_id:
                self.next_display_track_id += 1
            left, top, right, bottom = [int(value) for value in track.to_ltrb()]
            left = max(0, left)
            top = max(0, top)
            right = min(frame.shape[1] - 1, right)
            bottom = min(frame.shape[0] - 1, bottom)
            if right <= left or bottom <= top:
                continue

            center = ((left + right) // 2, (top + bottom) // 2)
            self.trails[track_id].append(center)
            self._draw_trail(frame, self.trails[track_id])
            self._draw_skeleton_overlay(frame, left, top, right, bottom, track_id, is_predicted)
            forecast_points = self._forecast_points(self.trails[track_id], frame.shape)
            if forecast_points:
                forecasted_track_ids.add(track_id)
                self._draw_forecast(frame, forecast_points, left, top, right, bottom)

            box_color = PREDICTED_COLOR if is_predicted else TRACK_COLOR
            line_style = cv2.LINE_AA
            thickness = 2 if is_predicted else 3
            self._draw_track_fill(frame, left, top, right, bottom, box_color, is_predicted)
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, thickness, line_style)
            cv2.circle(frame, center, 5 if is_predicted else 6, box_color, -1, cv2.LINE_AA)

            label_suffix = " (pred)" if is_predicted else ""
            label = f"Drone #{display_track_id}{label_suffix}"
            self._draw_label(frame, label, left, top, box_color)

        self.active_tracks = len(visible_track_ids)
        self.predicted_tracks = len(forecasted_track_ids)
        self._draw_status_panel(frame)
        return frame

    def _select_display_tracks(self, frame: np.ndarray, tracks) -> list:
        candidates = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            if track.time_since_update > self.config.max_predicted_frames:
                continue

            left, top, right, bottom = [int(value) for value in track.to_ltrb()]
            left = max(0, left)
            top = max(0, top)
            right = min(frame.shape[1] - 1, right)
            bottom = min(frame.shape[0] - 1, bottom)
            if right <= left or bottom <= top:
                continue

            area = max(1, (right - left) * (bottom - top))
            freshness_bonus = 1.0 if track.time_since_update == 0 else 0.6
            score = area * freshness_bonus
            candidates.append((score, track))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [candidates[0][1]]

    def _draw_trail(self, frame: np.ndarray, points: deque[tuple[int, int]]) -> None:
        if len(points) < 2:
            return

        for index in range(1, len(points)):
            fade = index / len(points)
            color = tuple(int(channel * fade) for channel in TRAIL_COLOR)
            thickness = max(2, int(2 + 4 * fade))
            cv2.line(frame, points[index - 1], points[index], color, thickness, cv2.LINE_AA)
            cv2.circle(frame, points[index], max(1, int(2 + 2 * fade)), color, -1, cv2.LINE_AA)

    def _draw_track_fill(
        self,
        frame: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
        color: tuple[int, int, int],
        is_predicted: bool,
    ) -> None:
        alpha = 0.08 if is_predicted else 0.12
        region = frame[top:bottom, left:right]
        if region.size == 0:
            return
        fill = np.full_like(region, BOX_FILL_COLOR if not is_predicted else (18, 52, 74))
        cv2.addWeighted(fill, alpha, region, 1 - alpha, 0, region)

    def _motion_fallback_detect(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        if self.previous_gray_frame is None:
            self.previous_gray_frame = gray
            return []

        diff = cv2.absdiff(self.previous_gray_frame, gray)
        self.previous_gray_frame = gray

        _, threshold = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        threshold = cv2.dilate(threshold, None, iterations=2)
        threshold = cv2.erode(threshold, None, iterations=1)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[tuple[float, Detection]] = []
        frame_height, frame_width = frame.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config.motion_min_area:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            if width < 16 or height < 16:
                continue
            if width > frame_width * 0.8 or height > frame_height * 0.8:
                continue

            detections.append(
                (
                    area,
                    Detection(
                    bbox_xywh=[float(x), float(y), float(width), float(height)],
                    confidence=0.35,
                    label="motion-drone",
                    ),
                )
            )

        if not detections:
            return []

        detections.sort(key=lambda item: item[0], reverse=True)
        strongest = [item[1] for item in detections[:1]]
        return self.detector._non_max_suppress(strongest, iou_threshold=0.35)

    def _draw_skeleton_overlay(
        self,
        frame: np.ndarray,
        left: int,
        top: int,
        right: int,
        bottom: int,
        track_id: int,
        is_predicted: bool,
    ) -> None:
        if is_predicted:
            return

        roi = frame[top:bottom, left:right]
        if roi.size == 0 or min(roi.shape[:2]) < 20:
            return

        cached = self.skeleton_cache.get(track_id)
        use_cached = (
            cached is not None
            and self.processed_frames - cached[0] < self.config.skeleton_refresh_interval
        )

        if use_cached:
            skeleton_mask = cached[1]
        else:
            skeleton_mask = self._extract_skeleton_mask(roi)
            if skeleton_mask is None:
                return
            self.skeleton_cache[track_id] = (self.processed_frames, skeleton_mask)

        if skeleton_mask.shape[:2] != roi.shape[:2]:
            skeleton_mask = cv2.resize(
                skeleton_mask,
                (roi.shape[1], roi.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        overlay = np.zeros_like(roi)
        overlay[skeleton_mask > 0] = SKELETON_COLOR
        glow = cv2.dilate(skeleton_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
        overlay[glow > 0] = np.maximum(overlay[glow > 0], np.array(SKELETON_GLOW, dtype=np.uint8))
        cv2.addWeighted(overlay, 0.6, roi, 1.0, 0, roi)

    def _extract_skeleton_mask(self, roi: np.ndarray) -> np.ndarray | None:
        height, width = roi.shape[:2]
        max_dim = max(height, width)
        scale = min(1.0, 96.0 / max_dim)
        sample = roi
        if scale < 1.0:
            sample = cv2.resize(
                roi,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted = cv2.bitwise_not(binary)

        candidates = [binary, inverted]
        chosen: np.ndarray | None = None
        best_score = float("-inf")

        for candidate in candidates:
            cleaned = cv2.morphologyEx(
                candidate,
                cv2.MORPH_OPEN,
                np.ones((3, 3), dtype=np.uint8),
                iterations=1,
            )
            filled_ratio = cv2.countNonZero(cleaned) / float(cleaned.shape[0] * cleaned.shape[1])
            if filled_ratio < 0.02 or filled_ratio > 0.55:
                continue

            edges = cv2.Canny(cleaned, 40, 140)
            score = cv2.countNonZero(edges) - abs(filled_ratio - 0.16) * 1000
            if score > best_score:
                best_score = score
                chosen = cleaned

        if chosen is None:
            return None

        skeleton = np.zeros_like(chosen)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        working = chosen.copy()

        for _ in range(18):
            eroded = cv2.erode(working, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(working, temp)
            skeleton = cv2.bitwise_or(skeleton, temp)
            working = eroded
            if cv2.countNonZero(working) == 0:
                break

        if cv2.countNonZero(skeleton) < 6:
            return None

        if skeleton.shape[:2] != roi.shape[:2]:
            skeleton = cv2.resize(
                skeleton,
                (roi.shape[1], roi.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        return skeleton

    def _forecast_points(
        self,
        points: deque[tuple[int, int]],
        frame_shape: tuple[int, ...],
    ) -> list[tuple[int, int]]:
        if len(points) < 2:
            return []

        recent_points = list(points)[-3:]
        deltas = [
            (
                recent_points[index][0] - recent_points[index - 1][0],
                recent_points[index][1] - recent_points[index - 1][1],
            )
            for index in range(1, len(recent_points))
        ]
        if not deltas:
            return []

        velocity_x = sum(delta[0] for delta in deltas) / len(deltas)
        velocity_y = sum(delta[1] for delta in deltas) / len(deltas)
        if abs(velocity_x) + abs(velocity_y) < 1.0:
            return []

        height, width = frame_shape[:2]
        origin_x, origin_y = recent_points[-1]
        forecast_points: list[tuple[int, int]] = []

        for step in range(1, self.config.forecast_steps + 1):
            next_x = int(round(origin_x + velocity_x * step))
            next_y = int(round(origin_y + velocity_y * step))
            next_x = min(width - 1, max(0, next_x))
            next_y = min(height - 1, max(0, next_y))
            forecast_points.append((next_x, next_y))

        return forecast_points

    def _draw_forecast(
        self,
        frame: np.ndarray,
        forecast_points: list[tuple[int, int]],
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> None:
        if not forecast_points:
            return

        current_center = ((left + right) // 2, (top + bottom) // 2)
        anchor = current_center

        cv2.arrowedLine(
            frame,
            current_center,
            forecast_points[0],
            FORECAST_COLOR,
            4,
            cv2.LINE_AA,
            tipLength=0.3,
        )

        for index, point in enumerate(forecast_points, start=1):
            fade = 0.45 + (0.55 * (index / max(1, len(forecast_points))))
            color = tuple(min(255, int(channel * fade)) for channel in FORECAST_COLOR)
            cv2.line(frame, anchor, point, color, 4, cv2.LINE_AA)
            cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)
            anchor = point

        final_center = forecast_points[-1]
        half_width = max(10, (right - left) // 2)
        half_height = max(10, (bottom - top) // 2)
        ghost_left = max(0, final_center[0] - half_width)
        ghost_top = max(0, final_center[1] - half_height)
        ghost_right = min(frame.shape[1] - 1, final_center[0] + half_width)
        ghost_bottom = min(frame.shape[0] - 1, final_center[1] + half_height)

        cv2.rectangle(
            frame,
            (ghost_left, ghost_top),
            (ghost_right, ghost_bottom),
            FORECAST_COLOR,
            3,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            frame,
            (ghost_left, ghost_top),
            (ghost_right, ghost_bottom),
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Projected",
            (ghost_left, max(14, ghost_top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            FORECAST_COLOR,
            2,
            cv2.LINE_AA,
        )

    def _draw_label(self, frame: np.ndarray, text: str, left: int, top: int, color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.7
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, text_scale, thickness)
        text_top = max(0, top - text_height - baseline - 8)
        text_bottom = text_top + text_height + baseline + 8
        text_right = left + text_width + 12
        cv2.rectangle(frame, (left, text_top), (text_right, text_bottom), color, -1)
        cv2.putText(
            frame,
            text,
            (left + 6, text_bottom - baseline - 4),
            font,
            text_scale,
            TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_status_panel(self, frame: np.ndarray) -> None:
        panel_text = [
            f"Target labels: {', '.join(self.config.target_labels)}",
            f"Confidence >= {self.config.confidence_threshold:.2f}",
            f"Drone mode: {'on' if self.config.drone_mode else 'off'}",
            f"Detect every {self.config.detect_interval} frame(s)",
            "Overlay: box + trail + skeleton + projected path",
            "Fallback: motion assist when YOLO misses",
            f"Tracks: {self.active_tracks} active / {len(self.seen_track_ids)} total",
            f"Forecasts: {self.predicted_tracks}",
            f"FPS: {self.fps:.2f} proc / {self.source_fps:.2f} src",
        ]

        origin_x, origin_y = 16, 24
        line_height = 24
        panel_height = line_height * len(panel_text) + 18
        panel_width = 430
        cv2.rectangle(frame, (origin_x - 10, origin_y - 18), (origin_x + panel_width, origin_y + panel_height), (10, 18, 28), -1)
        for index, text in enumerate(panel_text):
            cv2.putText(
                frame,
                text,
                (origin_x, origin_y + index * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (230, 240, 245),
                2,
                cv2.LINE_AA,
            )


class SessionManager:
    def __init__(self) -> None:
        self.detector_factory = DetectorFactory()
        self.sessions: dict[str, VideoTrackingSession] = {}
        self.upload_dir = Path(__file__).resolve().parent / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, file_name: str, file_bytes: bytes, config: TrackingConfig) -> VideoTrackingSession:
        detector = self.detector_factory.get()
        session_id = uuid.uuid4().hex
        safe_name = Path(file_name).name or f"{session_id}.mp4"
        video_path = self.upload_dir / f"{session_id}-{safe_name}"
        video_path.write_bytes(file_bytes)

        session = VideoTrackingSession(
            session_id=session_id,
            video_path=video_path,
            detector=detector,
            config=config,
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> VideoTrackingSession | None:
        return self.sessions.get(session_id)
