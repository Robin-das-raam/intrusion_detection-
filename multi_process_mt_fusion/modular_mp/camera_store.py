# camera_store.py
# Simple in-memory camera store

from datetime import datetime
import uuid

# { cam_id: { id, name, rtsp_url, location, status, fps, addedAt } }
_cameras = {}

def generate_id():
    return "cam_" + uuid.uuid4().hex[:8]

def add_camera(name: str, rtsp_url: str, location: str = "", status: str = "online") -> dict:
    cam_id = generate_id()
    camera = {
        "id": cam_id,
        "name": name,
        "rtsp_url": rtsp_url,
        "location": location,
        "status": status,
        "fps": 25,
        "zoneCount": 0,
        "addedAt": datetime.utcnow().isoformat() + "Z"
    }
    _cameras[cam_id] = camera
    return camera

def get_all_cameras() -> list:
    return list(_cameras.values())

def get_camera(cam_id: str) -> dict | None:
    return _cameras.get(cam_id)

def delete_camera(cam_id: str) -> bool:
    if cam_id in _cameras:
        del _cameras[cam_id]
        return True
    return False

def update_camera_status(cam_id: str, status: str):
    if cam_id in _cameras:
        _cameras[cam_id]["status"] = status