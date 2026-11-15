from __future__ import annotations

import os
import hashlib
import math
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from .cache import file_digest, load_cache, save_cache

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
DATA_DIR = ROOT / "uploads"
CACHE_DIR = ROOT / "analysis_cache"
DATA_DIR.mkdir(exist_ok=True)
SAMPLE_FPS = min(10.0, max(1.0, float(os.getenv("SAMPLE_FPS", "4"))))
THRESHOLD = min(1.0, max(0.0, float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "500")) * 1024 * 1024
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "real-time-accident-detection/1")


def cache_settings():
    return {'model_id': MODEL_ID, 'api_url': os.getenv('ROBOFLOW_API_URL', 'https://serverless.roboflow.com'),
            'sample_fps': SAMPLE_FPS, 'threshold': THRESHOLD, 'analysis_version': 1}


def new_job():
    return {'status': 'analyzing', 'progress': 0, 'processed_frames': 0, 'results': [],
            'events': [], 'error': None, 'duration': 0, 'source': 'roboflow', 'cached_at': None}

app = FastAPI(title="TrafficGuard AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

videos: dict[str, dict[str, Any]] = {}
jobs: dict[str, dict[str, Any]] = {}
lock = threading.Lock()

ACCIDENT_TYPES = {
    "car_car_accident": ("vehicle_vehicle", "Va chạm giữa các ô tô"),
    "car_bike_accident": ("motorcycle", "Tai nạn liên quan đến xe máy"),
    "bike_bike_accident": ("motorcycle", "Tai nạn liên quan đến xe máy"),
    "bike_object_accident": ("motorcycle", "Tai nạn liên quan đến xe máy"),
    "car_object_accident": ("single_vehicle", "Xe va chạm với vật thể"),
    "car_person_accident": ("pedestrian", "Va chạm liên quan đến người đi bộ"),
    "bike_person_accident": ("pedestrian", "Va chạm liên quan đến người đi bộ"),
}


def classify(predictions: list[dict[str, Any]], threshold: float = THRESHOLD) -> dict[str, Any]:
    """Return all accident labels above threshold and the strongest primary label."""
    accidents = []
    for pred in predictions:
        label = str(pred.get("class", pred.get("class_name", ""))).strip().lower()
        try:
            confidence = float(pred.get("confidence", 0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            continue
        if label in ACCIDENT_TYPES and confidence >= threshold:
            category, title = ACCIDENT_TYPES[label]
            class_id = {'vehicle_vehicle': 1, 'motorcycle': 2, 'single_vehicle': 3, 'pedestrian': 4}[category]
            accidents.append({"label": label, "type": category, "class_id": class_id, "title": title, "confidence": confidence})
    accidents.sort(key=lambda item: item["confidence"], reverse=True)
    return {"status": "accident" if accidents else "normal", "accidents": accidents,
            "primary": accidents[0] if accidents else None}


def cached_analysis(digest: str):
    """Rebuild classification from genuine stored predictions using current mapping logic."""
    job = load_cache(CACHE_DIR, digest, cache_settings())
    if job is None:
        return None
    events = []
    for frame in job['results']:
        frame['classification'] = classify(frame['predictions'])
        for accident in frame['classification']['accidents']:
            record_event(events, accident, frame['timestamp'])
    job['events'] = events
    return job


def box_to_canvas(prediction: dict[str, float], video_width: float, video_height: float,
                  canvas_width: float, canvas_height: float) -> dict[str, float]:
    """Map Roboflow center-based source pixels into an object-fit:contain canvas."""
    scale = min(canvas_width / video_width, canvas_height / video_height)
    offset_x = (canvas_width - video_width * scale) / 2
    offset_y = (canvas_height - video_height * scale) / 2
    return {"x": offset_x + (prediction["x"] - prediction["width"] / 2) * scale,
            "y": offset_y + (prediction["y"] - prediction["height"] / 2) * scale,
            "width": prediction["width"] * scale, "height": prediction["height"] * scale}


def record_event(events: list[dict[str, Any]], accident: dict[str, Any], timestamp: float,
                 cooldown: float = 3.0) -> None:
    """Merge consecutive same-type alarms during the cooldown, retaining peak confidence."""
    previous = next((event for event in reversed(events)
                     if event["type"] == accident["type"] and timestamp - event["last_seen"] <= cooldown), None)
    if previous:
        previous["last_seen"] = timestamp
        previous["confidence"] = max(previous["confidence"], accident["confidence"])
        if accident["label"] not in previous["labels"]:
            previous["labels"].append(accident["label"])
    else:
        events.append({"id": str(uuid.uuid4()), "timestamp": timestamp, "last_seen": timestamp,
                       "type": accident["type"], "title": accident["title"],
                       "confidence": accident["confidence"], "labels": [accident["label"]]})


def validate_video_filename(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(400, "Unsupported video format. Upload MP4, MOV, AVI, MKV, WEBM, or M4V.")
    return suffix


def write_upload_chunk(out, chunk: bytes, current_size: int) -> int:
    next_size = current_size + len(chunk)
    if next_size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    out.write(chunk)
    return next_size


def _client():
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not key or key == "your_api_key":
        raise RuntimeError("ROBOFLOW_API_KEY is missing. Add it to backend/.env to run inference.")
    from inference_sdk import InferenceHTTPClient
    return InferenceHTTPClient(api_url=os.getenv("ROBOFLOW_API_URL", "https://serverless.roboflow.com"), api_key=key)


def _run_analysis(video_id: str) -> None:
    record = videos[video_id]
    job = jobs[video_id]
    cap = None
    try:
        client = _client()
        cap = cv2.VideoCapture(str(record["path"]))
        if not cap.isOpened():
            raise ValueError("OpenCV could not read this video. Try a standard MP4 (H.264) file.")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / source_fps if total_frames else 0
        stride = max(1, round(source_fps / SAMPLE_FPS))
        result_frames: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        # Keep all raw frame predictions; group adjacent same-type alarms within 3 seconds.
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % stride == 0:
                timestamp = frame_index / source_fps
                response = client.infer(frame, model_id=MODEL_ID)
                if not isinstance(response, dict) or not isinstance(response.get('predictions'), list):
                    raise ValueError('Roboflow returned an invalid detection response; frame status is unknown.')
                predictions = response['predictions']
                normalized = []
                for p in predictions:
                    try:
                        confidence = float(p.get("confidence", 0))
                        normalized.append({
                            "class": str(p.get("class", p.get("class_name", "unknown"))),
                            "confidence": confidence,
                            "x": float(p["x"]), "y": float(p["y"]),
                            "width": float(p["width"]), "height": float(p["height"]),
                        })
                    except (KeyError, TypeError, ValueError):
                        continue
                classification = classify(normalized)
                result_frames.append({"frame_index": frame_index, "timestamp": timestamp,
                                      "predictions": normalized, "classification": classification})
                for accident in classification["accidents"]:
                    record_event(events, accident, timestamp)
                with lock:
                    job["progress"] = min(99, round(frame_index / max(total_frames, 1) * 100))
                    job["processed_frames"] = len(result_frames)
                    job["results"] = result_frames
                    job["events"] = events
            frame_index += 1
        cap.release()
        if not result_frames:
            raise ValueError("No frames were available to analyze.")
        with lock:
            job.update(status="completed", progress=100, duration=duration,
                       processed_frames=len(result_frames), results=result_frames, events=events)
        try:
            digest = record.get('sha256') or file_digest(record['path'])
            record['sha256'] = digest
            job['cached_at'] = save_cache(CACHE_DIR, digest, cache_settings(), job)
        except OSError:
            job['cache_warning'] = 'Analysis completed, but results could not be saved to disk.'
    except Exception as exc:
        if cap is not None:
            cap.release()
        with lock:
            message = str(exc)
            key = os.getenv('ROBOFLOW_API_KEY', '')
            if key:
                message = message.replace(key, '[redacted]')
            job.update(status="error", error=message, progress=0)


@app.get("/api/health")
def health():
    key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    return {"status": "ok", "model_id": MODEL_ID,
            "api_key_configured": bool(key and key != "your_api_key"),
            "confidence_threshold": THRESHOLD, "sample_fps": SAMPLE_FPS}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    suffix = validate_video_filename(file.filename or "")
    video_id = str(uuid.uuid4())
    path = DATA_DIR / f"{video_id}{suffix}"
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size = write_upload_chunk(out, chunk, size)
                digest.update(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    videos[video_id] = {"path": path, "filename": Path(file.filename or "video").name, "size": size,
                        'sha256': digest.hexdigest()}
    cached = cached_analysis(digest.hexdigest())
    return {"video_id": video_id, "filename": videos[video_id]["filename"], "size": size,
            "video_url": f"/api/video/{video_id}", 'cache_available': cached is not None}


@app.post("/api/analyze/{video_id}")
def analyze_video(video_id: str, background_tasks: BackgroundTasks, force: bool = False):
    if video_id not in videos:
        raise HTTPException(404, "Video not found")
    if video_id in jobs and jobs[video_id]["status"] == "analyzing":
        return {"status": "analyzing", "video_id": video_id}
    record = videos[video_id]
    digest = record.get('sha256') or file_digest(record['path'])
    record['sha256'] = digest
    cached = None if force else cached_analysis(digest)
    if cached:
        jobs[video_id] = cached
        return {'status': 'completed', 'video_id': video_id, 'source': 'cache', 'cached_at': cached['cached_at']}
    jobs[video_id] = new_job()
    background_tasks.add_task(_run_analysis, video_id)
    return {"status": "analyzing", "video_id": video_id}


@app.get("/api/analyze/{video_id}/status")
def analysis_status(video_id: str):
    if video_id not in jobs:
        raise HTTPException(404, "Analysis not started")
    job = jobs[video_id]
    return {key: job.get(key) for key in ("status", "progress", "processed_frames", "error", "duration", 'source', 'cached_at', 'cache_warning')}


@app.get("/api/analyze/{video_id}/results")
def analysis_results(video_id: str):
    if video_id not in jobs:
        raise HTTPException(404, "Analysis not started")
    job = jobs[video_id]
    return {"status": job["status"], "results": job["results"], "events": job["events"], "error": job["error"],
            'source': job.get('source', 'roboflow'), 'cached_at': job.get('cached_at')}


@app.get("/api/video/{video_id}")
def serve_video(video_id: str):
    record = videos.get(video_id)
    if not record or not record["path"].is_file():
        raise HTTPException(404, "Video not found")
    return FileResponse(record["path"], media_type="video/mp4", filename=record["filename"], content_disposition_type="inline")
