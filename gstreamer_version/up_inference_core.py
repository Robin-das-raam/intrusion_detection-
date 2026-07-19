# up_inference_core.py

import os
import json
import numpy as np
import cv2
from datetime import datetime

def load_zones(zpath):
    """
    Robustly loads zone data from a JSON file.
    Handles both a direct list `[...]` and a dictionary `{"zones": [...]}`.
    """
    if not os.path.exists(zpath):
        print(f"[Warning] Zone file not found: {zpath}")
        return []

    with open(zpath, "r") as f:
        data = json.load(f)

    # This handles both JSON formats gracefully
    raw = data if isinstance(data, list) else data.get("zones", [])
    zones = []

    for z in raw:
        zones.append({
            "name": z.get("name", "zone"),
            "points": z.get("points", [])
        })

    return zones

# def scale_zones(zones, src_shape, dst_shape):
#     """
#     Scales zone coordinates from a source shape to a destination shape.
#     """
#     src_w, src_h = src_shape
#     dst_w, dst_h = dst_shape

#     scaled = []
#     for z in zones:
#         pts = []
#         # Normalize coordinates (0-1) and then scale to destination
#         for x, y in z["points"]:
#             nx = int((x / src_w) * dst_w)
#             ny = int((y / src_h) * dst_h)
#             pts.append((nx, ny))

#         scaled.append({
#             "name": z["name"],
#             "points": pts  # We will use this standardized key
#         })

#     return scaled

def scale_zones(zones, src_shape, dst_shape):
    """
    Scales zone coordinates from a source shape to a destination shape.
    src_shape / dst_shape are OpenCV shapes like (H, W, C) or (H, W).
    """
    src_h, src_w = src_shape[:2]   # [:2] takes only height, width
    dst_h, dst_w = dst_shape[:2]

    scaled = []
    for z in zones:
        pts = []
        for x, y in z["points"]:
            nx = int((x / src_w) * dst_w)
            ny = int((y / src_h) * dst_h)
            pts.append((nx, ny))

        scaled.append({
            "name": z["name"],
            "points": pts
        })

    return scaled

# def build_label_map(zones_scaled, dst_shape):
#     """
#     Creates a map where each pixel's value is the index of the zone it belongs to.
#     This version is robust against potential errors.
#     """
#     h, w = dst_shape
#     label_map = -np.ones((h, w), dtype=np.int16)

#     for idx, z in enumerate(zones_scaled):
#         # The 'points' key holds a list of tuples from scale_zones
#         pts_list = z.get("points", [])
        
#         # Check if we have enough points to draw a polygon
#         if len(pts_list) >= 3:
#             pts_array = np.array(pts_list, dtype=np.int32)
#             cv2.fillPoly(label_map, [pts_array], idx)
            
#     return label_map

# def build_zone_overlay(zones_scaled, dst_shape):
#     """Creates a transparent image with the zone outlines and names drawn on it."""
#     h, w = dst_shape
#     overlay = np.zeros((h, w, 3), dtype=np.uint8)
#     for z in zones_scaled:
#         pts_list = z.get("points", [])
#         if len(pts_list) >= 2:
#             pts_array = np.array(pts_list, dtype=np.int32)
#             cv2.polylines(overlay, [pts_array], True, (255, 100, 0), 2)
#             x0, y0 = pts_array[0]
#             cv2.putText(overlay, z["name"], (x0 + 5, y0 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
#     return overlay

def save_intrusion_snapshot(cam_id, frame, zone_name):
    """Saves a snapshot of the intrusion."""
    os.makedirs("alert_snapshots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_zone_name = "".join(c for c in zone_name if c.isalnum() or c in (' ', '_')).rstrip()
    path = f"alert_snapshots/cam{cam_id}_{safe_zone_name.replace(' ', '_')}_{ts}.jpg"
    cv2.imwrite(path, frame)
    return path

def build_label_map(zones_scaled, dst_shape):
    """
    Creates a map where each pixel's value is the index of the zone it belongs to.
    """
    h, w = dst_shape[:2]   # <-- FIX: [:2] ignores the 3rd channel dimension
    label_map = -np.ones((h, w), dtype=np.int16)

    for idx, z in enumerate(zones_scaled):
        pts_list = z.get("points", [])
        if len(pts_list) >= 3:
            pts_array = np.array(pts_list, dtype=np.int32)
            cv2.fillPoly(label_map, [pts_array], idx)
            
    return label_map


def build_zone_overlay(zones_scaled, dst_shape):
    """Creates a transparent image with the zone outlines and names drawn on it."""
    h, w = dst_shape[:2]   # <-- FIX: same here
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for z in zones_scaled:
        pts_list = z.get("points", [])
        if len(pts_list) >= 2:
            pts_array = np.array(pts_list, dtype=np.int32)
            cv2.polylines(overlay, [pts_array], True, (255, 100, 0), 2)
            x0, y0 = pts_array[0]
            cv2.putText(overlay, z["name"], (x0 + 5, y0 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
    return overlay