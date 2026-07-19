# zone_store.py
# In-memory zone storage

import uuid
from datetime import datetime

# { zone_id: { id, cameraId, name, points, enabled, createdAt } }
_zones = {}

def generate_id():
    return "zone_" + uuid.uuid4().hex[:8]

def add_zone(camera_id: str, name: str, points: list, enabled: bool = True) -> dict:
    zone_id = generate_id()
    zone = {
        "id": zone_id,
        "cameraId": camera_id,
        "name": name,
        "points": points,  # normalized (0-1) coordinates [{"x": 0.5, "y": 0.5}, ...]
        "enabled": enabled,
        "createdAt": datetime.utcnow().isoformat() + "Z"
    }
    _zones[zone_id] = zone
    return zone

def get_all_zones() -> list:
    return list(_zones.values())

def get_zones_by_camera(camera_id: str) -> list:
    return [z for z in _zones.values() if z["cameraId"] == camera_id]

def get_zone(zone_id: str) -> dict | None:
    return _zones.get(zone_id)

def delete_zone(zone_id: str) -> bool:
    if zone_id in _zones:
        del _zones[zone_id]
        return True
    return False

def delete_zones_by_camera(camera_id: str):
    """Delete all zones for a camera — call when camera is deleted."""
    to_delete = [zid for zid, z in _zones.items() if z["cameraId"] == camera_id]
    for zid in to_delete:
        del _zones[zid]

def toggle_zone(zone_id: str) -> dict | None:
    if zone_id in _zones:
        _zones[zone_id]["enabled"] = not _zones[zone_id]["enabled"]
        return _zones[zone_id]
    return None