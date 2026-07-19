# stream_manager.py
# Manages CameraThread instances per camera
# Reuses CameraThread from up_inference_core.py

import cv2
import threading
from up_inference_core import CameraThread

# { cam_id: CameraThread }
_streams = {}
_lock = threading.Lock()

def start_stream(cam_id: str, rtsp_url: str) -> bool:
    """Start a camera thread for given cam_id."""
    with _lock:
        if cam_id in _streams:
            return True  # already running
        try:
            cam = CameraThread(rtsp_url)
            _streams[cam_id] = cam
            return True
        except Exception as e:
            print(f"[StreamManager] Failed to start stream {cam_id}: {e}")
            return False

def stop_stream(cam_id: str) -> bool:
    """Stop and remove a camera thread."""
    with _lock:
        if cam_id not in _streams:
            return False
        try:
            _streams[cam_id].stop()
            del _streams[cam_id]
            return True
        except Exception as e:
            print(f"[StreamManager] Failed to stop stream {cam_id}: {e}")
            return False

def get_frame(cam_id: str):
    """Get latest frame from camera thread."""
    with _lock:
        cam = _streams.get(cam_id)
    if cam is None:
        return None
    return cam.read()

def is_running(cam_id: str) -> bool:
    return cam_id in _streams

def stop_all():
    """Stop all streams — call on shutdown."""
    with _lock:
        for cam_id, cam in _streams.items():
            try:
                cam.stop()
            except Exception:
                pass
        _streams.clear()