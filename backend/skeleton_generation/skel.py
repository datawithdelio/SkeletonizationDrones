import cv2 as cv
from ultralytics import YOLO
import numpy as np
import time
from multiprocessing import Process, Queue, Manager
import os
import torch

from skeleton_generation.utils.skeleton.extractKimiaEDF import generate_skeleton
from skeleton_generation.utils.processing_utils.create_overlay import overlay_images
from skeleton_generation.utils.processing_utils.process_images import process_image


current_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_directory, "utils", "models", "yolov8n-seg.onnx")


def filter_most_confident(detections):
    max_confidence = 0
    best_detection = None

    for detection in detections:
        if detection.boxes.conf.numel() > 0:
            confidence = detection.boxes.conf[0]
            if confidence >= max_confidence:
                max_confidence = confidence
                best_detection = detection

    return best_detection


def frame_reader(input_path, frame_queue, num_workers):
    video = cv.VideoCapture(input_path)
    assert video.isOpened(), "Error: Cannot Open Video!"

    frame_index = 0
    while True:
        ret, frame = video.read()
        if not ret:
            for _ in range(num_workers):
                frame_queue.put(None)
            break
        frame_queue.put((frame_index, frame))
        frame_index += 1

    video.release()


def process_frame(frame_index, frame, model, generation_settings, points_data):
    results = model.predict(frame, conf=generation_settings['confidence_level'], save=False, show=False, verbose=False)

    background = frame
    frame_results = []

    if results is None or len(results) == 0:
        frame_results.append(frame)
        return (frame_index, frame_results)

    produced_overlay = False

    for result in results:
        if result is None or result.masks is None or len(result) == 0:
            continue

        img = np.copy(result.orig_img)

        for ci, c in enumerate(result):
            if c.masks is None or not c.masks.xy:
                continue

            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = np.array(c.masks.xy[0], dtype=np.int32).reshape(-1, 1, 2)
            _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)

            mask3ch = cv.cvtColor(b_mask, cv.COLOR_GRAY2BGR)
            white_inside_crop = np.ones_like(img) * 255
            isolated = cv.bitwise_and(mask3ch, white_inside_crop)

            processed_img = process_image(isolated)
            if not processed_img["contour_strings"]:
                continue

            skel = generate_skeleton(
                processed_img["contour_strings"],
                frame.shape[1],
                frame.shape[0],
                generation_settings['smoothing_factor'],
                generation_settings['downsample'],
                points_data,
            )
            overlayed = overlay_images(background, skel)
            if overlayed is None:
                continue

            if overlayed.shape[2] == 4:
                overlayed = cv.cvtColor(overlayed, cv.COLOR_BGRA2BGR)

            if ci != (len(result.boxes) - 1):
                background = overlayed
            else:
                frame_results.append(overlayed)
                produced_overlay = True

    if not produced_overlay:
        frame_results.append(frame)

    return (frame_index, frame_results)


def process_frame_single_detection(frame_index, frame, model, generation_settings):
    results = model.predict(frame, conf=generation_settings['confidence_level'], save=False, show=False, verbose=False)

    background = frame
    frame_results = []

    best_result = filter_most_confident(results)

    if best_result is not None:
        img = np.copy(best_result.orig_img)

        for c in best_result:
            if c.masks is None or not c.masks.xy:
                continue

            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = np.array(c.masks.xy[0], dtype=np.int32).reshape(-1, 1, 2)
            _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)
            mask3ch = cv.cvtColor(b_mask, cv.COLOR_GRAY2BGR)
            isolated = cv.bitwise_and(mask3ch, img)
            processed_img = process_image(isolated)
            if not processed_img["contour_strings"]:
                continue

            skel = generate_skeleton(
                processed_img["contour_strings"],
                frame.shape[1],
                frame.shape[0],
                generation_settings['smoothing_factor'],
                generation_settings['downsample'],
                [],
            )
            overlayed = overlay_images(background, skel)
            if overlayed is None:
                continue

            if overlayed.shape[2] == 4:
                overlayed = cv.cvtColor(overlayed, cv.COLOR_BGRA2BGR)

            frame_results.append(overlayed)

    return (frame_index, frame_results)


def worker(frame_queue, result_queue, model_path, generation_settings, points_data):
    model = YOLO(model_path)
    while True:
        frame_data = frame_queue.get()
        if frame_data is None:
            result_queue.put(None)
            break
        frame_index, frame = frame_data
        results = process_frame(frame_index, frame, model, generation_settings, points_data)
        result_queue.put(results)


def video_writer(output_path, result_queue, frame_count, width, height, fps):
    video_writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"avc1"), fps, (width, height))

    frames = [None] * frame_count
    finished_frames = 0

    while finished_frames < frame_count:
        frame_data = result_queue.get()
        if frame_data is None:
            continue
        frame_index, frame_results = frame_data
        if frames[frame_index] is None:
            frames[frame_index] = frame_results[0] if frame_results else np.zeros((height, width, 3), dtype=np.uint8)
            finished_frames += 1

    for frame in frames:
        if frame is not None:
            video_writer.write(frame)

    video_writer.release()


def skeletonize_video(input_path, output_path, file_name, skeleton_data_name, generation_settings):
    model = model_path

    video = cv.VideoCapture(input_path)
    assert video.isOpened(), "Error: Cannot Open Video!"

    frame_count = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    video.release()

    frame_queue = Queue()
    result_queue = Queue()

    start_time = time.monotonic()

    manager = Manager()
    points_data = manager.list()

    num_workers = 9
    reader_process = Process(target=frame_reader, args=(input_path, frame_queue, num_workers))
    reader_process.start()

    workers = []
    for _ in range(num_workers):
        worker_process = Process(target=worker, args=(frame_queue, result_queue, model, generation_settings, points_data))
        workers.append(worker_process)
        worker_process.start()

    writer_process = Process(
        target=video_writer,
        args=(os.path.join(output_path, file_name), result_queue, frame_count, width, height, fps),
    )
    writer_process.start()

    reader_process.join()
    for w in workers:
        w.join()
    result_queue.put(None)
    writer_process.join()

    points_data = list(points_data)
    torch.save(points_data, os.path.join(output_path, skeleton_data_name))

    print(f"Finished in {time.monotonic() - start_time} seconds")
    cv.destroyAllWindows()


def _contour_strings_from_binary_mask(mask):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    min_area = max(50, int(mask.shape[0] * mask.shape[1] * 0.0005))
    filtered = [cnt for cnt in contours if cv.contourArea(cnt) >= min_area]
    if not filtered:
        return []

    height = mask.shape[0]
    contour_strings = [
        "{:.7e} {:.7e}".format(float(point[0][0]), float(height - point[0][1]))
        for contour in filtered
        for point in contour
    ]
    return contour_strings


def _build_fallback_contour_strings(input_path, original_img):
    original_h, original_w = original_img.shape[:2]

    raw = cv.imread(input_path, cv.IMREAD_UNCHANGED)
    if raw is not None and len(raw.shape) == 3 and raw.shape[2] == 4:
        alpha = raw[:, :, 3]
        if alpha.shape[:2] != (original_h, original_w):
            alpha = cv.resize(alpha, (original_w, original_h), interpolation=cv.INTER_NEAREST)
        _, mask = cv.threshold(alpha, 0, 255, cv.THRESH_BINARY)
    else:
        gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
        _, otsu = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        border = np.concatenate((otsu[0, :], otsu[-1, :], otsu[:, 0], otsu[:, -1]))
        border_is_bright = np.mean(border) > 127
        mask = cv.bitwise_not(otsu) if border_is_bright else otsu

    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)

    return _contour_strings_from_binary_mask(mask)


def skeletonize_img(input_path, output_path, file_name, skeleton_data_name, generation_settings):
    original_img = cv.imread(input_path)
    if original_img is None:
        return False

    model = YOLO(model_path)
    results = model.predict(original_img, conf=generation_settings['confidence_level'], save=False, show=False, verbose=False)

    points_data = []
    background = original_img
    wrote_output = False

    for result in results or []:
        if result is None or result.masks is None or len(result) == 0:
            continue

        img = np.copy(result.orig_img)

        for ci, c in enumerate(result):
            if c.masks is None or not c.masks.xy:
                continue

            b_mask = np.zeros(img.shape[:2], np.uint8)
            contour = np.array(c.masks.xy[0], dtype=np.int32).reshape(-1, 1, 2)
            _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)

            mask3ch = cv.cvtColor(b_mask, cv.COLOR_GRAY2BGR)
            white_inside_crop = np.ones_like(img) * 255
            isolated = cv.bitwise_and(mask3ch, white_inside_crop)

            processed_img = process_image(isolated)
            if not processed_img["contour_strings"]:
                continue

            skel = generate_skeleton(
                processed_img["contour_strings"],
                original_img.shape[1],
                original_img.shape[0],
                generation_settings['smoothing_factor'],
                generation_settings['downsample'],
                points_data,
            )
            overlayed = overlay_images(background, skel)
            if overlayed is None:
                continue

            if overlayed.shape[2] == 4:
                overlayed = cv.cvtColor(overlayed, cv.COLOR_BGRA2BGR)

            if ci != (len(result.boxes) - 1):
                background = overlayed
            else:
                cv.imwrite(os.path.join(output_path, file_name), overlayed)
                wrote_output = True

    if not wrote_output:
        contour_strings = _build_fallback_contour_strings(input_path, original_img)
        if contour_strings:
            skel = generate_skeleton(
                contour_strings,
                original_img.shape[1],
                original_img.shape[0],
                generation_settings['smoothing_factor'],
                generation_settings['downsample'],
                points_data,
            )
            overlayed = overlay_images(original_img, skel)
            if overlayed is not None:
                if overlayed.shape[2] == 4:
                    overlayed = cv.cvtColor(overlayed, cv.COLOR_BGRA2BGR)
                cv.imwrite(os.path.join(output_path, file_name), overlayed)
                wrote_output = True

    torch.save(points_data, os.path.join(output_path, skeleton_data_name))
    return wrote_output


def skeletonize_img_single_detection(input_path, output_path, file_name, generation_settings):
    original_img = cv.imread(input_path)

    model = YOLO(model_path)

    results = model.predict(original_img, conf=generation_settings['confidence_level'], save=False, show=False, verbose=False)

    background = original_img

    for result in results:
        if result is not None:
            img = np.copy(result.orig_img)

            for ci, c in enumerate(result):
                if c.masks is None or not c.masks.xy:
                    continue
                b_mask = np.zeros(img.shape[:2], np.uint8)
                contour = np.array(c.masks.xy[0], dtype=np.int32).reshape(-1, 1, 2)
                _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)
                mask3ch = cv.cvtColor(b_mask, cv.COLOR_GRAY2BGR)
                isolated = cv.bitwise_and(mask3ch, img)
                processed_img = process_image(isolated)
                if not processed_img["contour_strings"]:
                    continue
                skel = generate_skeleton(
                    processed_img["contour_strings"],
                    original_img.shape[1],
                    original_img.shape[0],
                    generation_settings['smoothing_factor'],
                    generation_settings['downsample'],
                    [],
                )
                overlayed = overlay_images(background, skel)

                if overlayed is None:
                    continue
                if overlayed.shape[2] == 4:
                    overlayed = cv.cvtColor(overlayed, cv.COLOR_BGRA2BGR)

                if ci != (len(results[0].boxes) - 1):
                    background = overlayed
                else:
                    cv.imwrite((f"{output_path}\\{file_name}"), overlayed)
        else:
            cv.imwrite((f"{output_path}\\{file_name}"), original_img)
