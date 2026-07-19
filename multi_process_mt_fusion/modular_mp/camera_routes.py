# camera_routes.py

import cv2
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from camera_store import (
    add_camera, get_all_cameras,
    get_camera, delete_camera,
    update_camera_status
)
from stream_manager import start_stream, stop_stream, get_frame, is_running

router = APIRouter(prefix="/api")


# ─────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────

class CameraCreate(BaseModel):
    name: str
    rtsp_url: str
    location: str = ""
    status: str = "online"


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@router.get("/cameras")
def list_cameras():
    """Return all registered cameras."""
    return get_all_cameras()


@router.post("/cameras", status_code=201)
def register_camera(body: CameraCreate):
    """
    Register a new camera and start its live stream.
    Called when user clicks 'Add Camera' in frontend.
    """
    # Save to store
    camera = add_camera(
        name=body.name,
        rtsp_url=body.rtsp_url,
        location=body.location,
        status=body.status
    )

    # Start stream thread
    success = start_stream(camera["id"], body.rtsp_url)

    if not success:
        # Still save camera but mark as error
        update_camera_status(camera["id"], "error")
        camera["status"] = "error"

    return camera


@router.delete("/cameras/{cam_id}")
def remove_camera(cam_id: str):
    """
    Stop stream and remove camera.
    Called when user clicks 'Delete' in frontend.
    """
    camera = get_camera(cam_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    stop_stream(cam_id)
    delete_camera(cam_id)

    return {"message": f"Camera {cam_id} deleted successfully"}


@router.get("/stream/{cam_id}")
def stream_camera(cam_id: str):
    """
    MJPEG stream for a single camera.
    Use in frontend as: <img src="http://localhost:8000/api/stream/{cam_id}" />
    """
    camera = get_camera(cam_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not is_running(cam_id):
        # Try to restart if not running
        success = start_stream(cam_id, camera["rtsp_url"])
        if not success:
            raise HTTPException(status_code=503, detail="Stream unavailable")

    def generate():
        while True:
            frame = get_frame(cam_id)

            if frame is None:
                time.sleep(0.05)
                continue

            # Encode to JPEG
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