# inference_routes.py
# Inference API endpoints
# Handles start/stop/status and MJPEG streams for AI inference

import cv2
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from inference.engine import (
    start_inference,
    stop_inference,
    is_running,
    get_inference_frame,
    get_all_inference_cam_ids,
    get_engine_status
)

router = APIRouter(prefix="/api/inference")


# ─────────────────────────────────────────
# Control endpoints
# ─────────────────────────────────────────

@router.post("/start")
def start():
    """
    Start AI inference engine.
    Called when user clicks 'Start Inference' in Live View.
    """
    result = start_inference()
    return result


@router.post("/stop")
def stop():
    """
    Stop AI inference engine.
    Called when user clicks 'Stop Inference' in Live View.
    """
    result = stop_inference()
    return result


@router.get("/status")
def status():
    """
    Get inference engine status.
    Returns running state and active camera ids.
    """
    return get_engine_status()


# ─────────────────────────────────────────
# Stream endpoints
# ─────────────────────────────────────────

@router.get("/stream/{cam_id}")
def stream_inference(cam_id: str):
    """
    MJPEG stream with AI inference annotations for a single camera.
    Use in frontend as:
    <img src="http://localhost:8000/api/inference/stream/{cam_id}" />
    """
    if not is_running():
        raise HTTPException(
            status_code=400,
            detail="Inference is not running. Start inference first."
        )

    def generate():
        while is_running():
            frame = get_inference_frame(cam_id)

            if frame is None:
                time.sleep(0.05)
                continue

            ret, buffer = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )

            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

            time.sleep(0.033)  # ~30 FPS

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )