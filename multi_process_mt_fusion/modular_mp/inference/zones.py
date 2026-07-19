# inference/zones.py
# Zone utilities
# Handles zone scaling, label map and overlay building

import cv2
import numpy as np


def scale_zones_from_normalized(zones: list, dst_shape: tuple) -> list:
    """
    Convert normalized (0-1) zone points to pixel coordinates.

    Args:
        zones     : list of zone dicts from zone_store
                    each zone has {"name": str, "points": [{"x": 0-1, "y": 0-1}]}
        dst_shape : (H, W) or (H, W, C) of the target frame

    Returns:
        list of scaled zone dicts with pixel coordinates
        each zone has {"name": str, "points": [(px, py), ...]}
    """
    dst_h, dst_w = dst_shape[:2]
    scaled = []
    for z in zones:
        pts = []
        for p in z["points"]:
            px = int(p["x"] * dst_w)
            py = int(p["y"] * dst_h)
            pts.append((px, py))
        scaled.append({
            "name": z["name"],
            "points": pts
        })
    return scaled


def build_label_map(zones_scaled: list, dst_shape: tuple) -> np.ndarray:
    """
    Build a pixel label map where each pixel value is
    the index of the zone it belongs to (-1 if no zone).

    Args:
        zones_scaled : list of scaled zone dicts with pixel coordinates
        dst_shape    : (H, W) or (H, W, C) of the target frame

    Returns:
        np.ndarray: label map of shape (H, W) with int16 values
    """
    h, w = dst_shape[:2]
    label_map = -np.ones((h, w), dtype=np.int16)

    for idx, z in enumerate(zones_scaled):
        pts_list = z.get("points", [])
        if len(pts_list) >= 3:
            pts_array = np.array(pts_list, dtype=np.int32)
            cv2.fillPoly(label_map, [pts_array], idx)

    return label_map


def build_zone_overlay(zones_scaled: list, dst_shape: tuple) -> np.ndarray:
    """
    Build a transparent overlay image with zone outlines and names.

    Args:
        zones_scaled : list of scaled zone dicts with pixel coordinates
        dst_shape    : (H, W) or (H, W, C) of the target frame

    Returns:
        np.ndarray: overlay image of shape (H, W, 3)
    """
    h, w = dst_shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)

    for z in zones_scaled:
        pts_list = z.get("points", [])
        if len(pts_list) >= 2:
            pts_array = np.array(pts_list, dtype=np.int32)
            cv2.polylines(overlay, [pts_array], True, (255, 100, 0), 2)
            x0, y0 = pts_array[0]
            cv2.putText(
                overlay,
                z["name"],
                (x0 + 5, y0 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 100, 0),
                2
            )

    return overlay


def get_intruded_zone(
    feet_x: int,
    feet_y: int,
    label_map: np.ndarray,
    zones_scaled: list,
    frame_shape: tuple
) -> str | None:
    """
    Check if a person's feet position is inside any zone.

    Args:
        feet_x      : x coordinate of feet
        feet_y      : y coordinate of feet
        label_map   : pixel label map from build_label_map
        zones_scaled: list of scaled zone dicts
        frame_shape : (H, W) or (H, W, C) of the frame

    Returns:
        str | None: zone name if intruded, None otherwise
    """
    h, w = frame_shape[:2]

    if 0 <= feet_x < w and 0 <= feet_y < h:
        zidx = int(label_map[feet_y, feet_x])
        if zidx != -1:
            return zones_scaled[zidx]["name"]

    return None