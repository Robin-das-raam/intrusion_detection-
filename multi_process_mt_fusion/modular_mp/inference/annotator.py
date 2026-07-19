# inference/annotator.py
# Frame annotation utilities
# Draws bounding boxes, labels, circles on frames

import cv2
import numpy as np
from inference.zones import get_intruded_zone


def annotate_frame(
    frame: np.ndarray,
    result,
    label_map: np.ndarray | None,
    zones_scaled: list | None,
    zone_overlay: np.ndarray | None,
) -> tuple[np.ndarray, list]:
    """
    Annotate a frame with detection boxes, labels and zone intrusions.

    Args:
        frame        : resized BGR frame
        result       : YOLOv8 result object
        label_map    : pixel label map from build_label_map (or None)
        zones_scaled : list of scaled zone dicts (or None)
        zone_overlay : overlay image with zone outlines (or None)

    Returns:
        tuple:
            - annotated frame (np.ndarray)
            - list of intrusion events [{"zone": str, "box": (x1,y1,x2,y2)}]
    """
    # Draw zone overlay on frame
    if zone_overlay is not None:
        annotated = cv2.addWeighted(frame, 1.0, zone_overlay, 1.0, 0.0)
    else:
        annotated = frame.copy()

    intrusions = []

    if result.boxes is None:
        return annotated, intrusions

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()

    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, box)
        feet_x = int((x1 + x2) / 2)
        feet_y = int(y2)

        # Check zone intrusion
        intruded_zone = None
        if label_map is not None and zones_scaled is not None:
            intruded_zone = get_intruded_zone(
                feet_x, feet_y,
                label_map, zones_scaled,
                annotated.shape
            )

        if intruded_zone:
            color = (0, 0, 255)
            label = f"INTRUSION: {intruded_zone}"
            intrusions.append({
                "zone": intruded_zone,
                "box": (x1, y1, x2, y2)
            })
        else:
            color = (0, 255, 0)
            label = "person"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw feet point
        cv2.circle(annotated, (feet_x, feet_y), 5, color, -1)

        # Draw label
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return annotated, intrusions


def draw_camera_label(frame: np.ndarray, cam_name: str) -> np.ndarray:
    """
    Draw camera name label on top left of frame.

    Args:
        frame    : BGR frame
        cam_name : camera name string

    Returns:
        np.ndarray: frame with label drawn
    """
    cv2.putText(
        frame,
        cam_name,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )
    return frame


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS on top right of frame.

    Args:
        frame : BGR frame
        fps   : frames per second value

    Returns:
        np.ndarray: frame with FPS drawn
    """
    h, w = frame.shape[:2]
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (w - 120, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )
    return frame