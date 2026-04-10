from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import torch

from tracking import SessionManager, parse_tracking_config


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="YOLO Drone Tracking System")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

session_manager = SessionManager()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def healthcheck() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "model": "yolov8n.pt",
            "device": "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"),
            "note": "The YOLO model lazy-loads on first upload. Base COCO YOLO still has no native drone class, so drone mode maps to airplane, bird, and kite.",
        }
    )


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    confidence_threshold: str = Form("0.25"),
    target_labels: str = Form("drone"),
    trail_length: str = Form("42"),
    resize_width: str = Form("640"),
    drone_mode: str = Form("false"),
    tile_stride: str = Form("6"),
    tile_threshold: str = Form("0.55"),
    max_age: str = Form("45"),
    n_init: str = Form("2"),
    nn_budget: str = Form("100"),
    max_predicted_frames: str = Form("18"),
    jpeg_quality: str = Form("80"),
) -> JSONResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A video filename is required.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
        raise HTTPException(status_code=400, detail="Upload an MP4, MOV, AVI, MKV, or WebM video.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    form_values = {
        "confidence_threshold": confidence_threshold,
        "target_labels": target_labels,
        "trail_length": trail_length,
        "resize_width": resize_width,
        "drone_mode": drone_mode,
        "tile_stride": tile_stride,
        "tile_threshold": tile_threshold,
        "max_age": max_age,
        "n_init": n_init,
        "nn_budget": nn_budget,
        "max_predicted_frames": max_predicted_frames,
        "jpeg_quality": jpeg_quality,
    }
    try:
        config = parse_tracking_config(form_values)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid numeric setting: {exc}") from exc

    try:
        session = session_manager.create_session(file.filename, file_bytes, config)
    except Exception as exc:  # pragma: no cover - startup/runtime guard
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return JSONResponse(
        {
            "session_id": session.session_id,
            "stream_url": f"/api/stream/{session.session_id}",
            "status_url": f"/api/status/{session.session_id}",
            "config": {
                "confidence_threshold": config.confidence_threshold,
                "target_labels": list(config.target_labels),
                "trail_length": config.trail_length,
                "resize_width": config.resize_width,
                "drone_mode": config.drone_mode,
                "tile_stride": config.tile_stride,
            },
            "note": "Base COCO YOLO does not include a drone class. Drone mode expands to airplane, bird, and kite and adds a tiled small-object pass.",
        }
    )


@app.get("/api/stream/{session_id}")
async def stream_video(session_id: str) -> StreamingResponse:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Tracking session not found.")

    return StreamingResponse(
        session.mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/status/{session_id}")
async def get_status(session_id: str) -> JSONResponse:
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Tracking session not found.")
    return JSONResponse(session.get_status())
