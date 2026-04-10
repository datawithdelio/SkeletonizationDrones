# YOLO Drone Detection And Tracking

This directory contains a self-contained FastAPI application that:

- uses `yolov8n.pt` from Ultralytics for per-frame detection
- filters YOLO detections by confidence before tracking
- passes those detections into DeepSORT for persistent IDs and short-gap re-identification
- streams annotated video frames back to the browser over MJPEG in real time
- overlays both persistent bounding boxes and fading motion trails

## Important Model Caveat

The base `yolov8n.pt` model is trained on COCO and **does not include a native `drone` class**.

To make it more drone-oriented while still using standard YOLO weights, this app now does two things by default:

- expands `drone` into the proxy labels `airplane,bird,kite`
- runs an extra tiled small-object pass every few frames to improve recall on distant drones

For production-grade drone detection, the next step is to fine-tune YOLO on a drone-specific dataset and keep the same tracking and streaming stack.

## Project Files

```text
detr_tracking_system/
├── app.py
├── tracking.py
├── requirements.txt
├── README.md
└── static/
    ├── app.js
    ├── index.html
    └── styles.css
```

## Setup

1. Open a terminal in this directory:

   ```bash
   cd /Users/deliorincon/Desktop/research/SkeletonizationDrones/detr_tracking_system
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the FastAPI server:

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. Open the app in your browser:

   ```text
   http://127.0.0.1:8000
   ```

6. Upload a drone video and click **Start Live Tracking**.

You should upload a video that actually contains drones if you want meaningful drone detection and tracking. You can upload another kind of video just to test that the app runs, but the results will only be useful if the footage contains a visible aerial target.

## How It Works

1. The browser uploads the source video to FastAPI.
2. The backend lazily loads `yolov8n.pt`.
3. Each frame is resized if needed, then YOLO runs inference.
4. In drone mode, the backend also runs a tiled pass on the frame every few frames to help find small distant drones.
5. Detections below the confidence threshold are discarded.
6. Remaining detections are passed into DeepSORT.
7. DeepSORT maintains a persistent track ID per drone candidate.
8. The backend draws:
   - `Drone #<id>` labels
   - tracked bounding boxes
   - fading trajectory trails over the last `N` frames
9. Frames are JPEG-encoded and streamed back to the browser over MJPEG.

## Performance Notes

- If CUDA is available, both YOLO and DeepSORT's appearance embedder use the GPU automatically.
- If Apple Silicon MPS is available, YOLO can run on `mps`.
- Lowering `Resize width` usually gives the biggest speedup for live rendering.
- The first run downloads YOLO weights, so it can take noticeably longer than later runs.

## Suggested Runtime Settings

- `Confidence threshold`: `0.80` to `0.90`
- `Resize width`: `640`
- `Drone mode`: `Off` for speed, `On` only for tiny distant drones
- `Tile pass every N frames`: `6` for speed, lower only if recall is too weak
- `DeepSORT max age`: `20` to `40`
- `Trail length`: `20` to `40`

## API Endpoints

- `GET /` serves the HTML frontend
- `GET /api/health` reports the loaded detector device
- `POST /api/upload` creates a tracking session from an uploaded video
- `GET /api/stream/{session_id}` streams annotated JPEG frames as MJPEG
- `GET /api/status/{session_id}` returns live processing statistics

## Troubleshooting

### The app says there is no drone class

That is expected for base COCO YOLO. Start with `drone` mode, then fine-tune YOLO later if you need a true drone category.

### Streaming is too slow

- reduce `Resize width`
- use a GPU if available
- raise the confidence threshold slightly to reduce tracker load

### A track ID changes after a long disappearance

DeepSORT helps with short re-identification gaps, but very long exits or drastic viewpoint changes can still force a new ID. Increasing `max_age` can help for short absences.
