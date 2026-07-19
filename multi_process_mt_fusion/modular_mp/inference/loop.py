# inference/loop.py
# Main inference loop
# Manages camera threads, motion detection, batched inference and annotation

import cv2
import math
import time
import threading
import numpy as np
from ultralytics import YOLO

from configs import MODEL_PATH, DEVICE, CONF_THRESH, RESIZED_DIM
from camera_store import get_all_cameras
from zone_store import get_zones_by_camera
from stream_manager import CameraThread

from inference.motion import (
    compute_motion_ratio,
    update_background,
    init_background
)
from inference.zones import (
    scale_zones_from_normalized,
    build_label_map,
    build_zone_overlay
)
from inference.annotator import annotate_frame, draw_camera_label


# ─────────────────────────────────────────
# Shared state
# ─────────────────────────────────────────
# { cam_id: annotated_frame }
inference_frames = {}
inference_frames_lock = threading.Lock()


# ─────────────────────────────────────────
# Utils
# ─────────────────────────────────────────
def round_up_to_stride(x, stride=32):
    return int(math.ceil(x / stride) * stride)


# ─────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────
def load_model():
    """Load and optimize YOLO model."""
    print(f"[Loop] Loading YOLO model on {DEVICE}...")
    model = YOLO(MODEL_PATH)

    if "cuda" in str(DEVICE).lower():
        try:
            model.to(DEVICE)
            model.fuse()
            model.model.half()
            print("[Loop] Model loaded on GPU with FP16")
        except Exception as e:
            print(f"[Loop] GPU optimization failed: {e}")
    else:
        print("[Loop] Model loaded on CPU")

    return model


# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
def run_inference_loop(stop_event: threading.Event):
    """
    Main inference loop.
    Reads from camera_store and zone_store dynamically.
    Stores annotated frames in inference_frames.

    Args:
        stop_event: threading.Event to signal stop
    """

    # -----------------------------
    # Load cameras
    # -----------------------------
    cameras = get_all_cameras()
    if not cameras:
        print("[Loop] No cameras registered. Stopping.")
        return

    print(f"[Loop] Found {len(cameras)} camera(s)")

    # -----------------------------
    # Start camera threads
    # -----------------------------
    cam_threads = {}  # cam_id -> CameraThread
    cam_names = {}    # cam_id -> camera name

    for cam in cameras:
        print(f"[Loop] Connecting: {cam['name']} ({cam['rtsp_url']})")
        try:
            cam_threads[cam["id"]] = CameraThread(cam["rtsp_url"])
            cam_names[cam["id"]] = cam["name"]
        except Exception as e:
            print(f"[Loop] Failed to connect {cam['id']}: {e}")

    if not cam_threads:
        print("[Loop] No cameras connected. Stopping.")
        return

    # -----------------------------
    # Resize dimensions
    # -----------------------------
    base_w, base_h = RESIZED_DIM
    w_r = round_up_to_stride(base_w, 32)
    h_r = round_up_to_stride(base_h, 32)
    resize_wh = (w_r, h_r)   # (W, H) for cv2.resize
    imgsz = (h_r, w_r)       # (H, W) for Ultralytics

    # -----------------------------
    # Per camera state
    # -----------------------------
    zones_scaled = {}       # cam_id -> scaled zones
    label_maps = {}         # cam_id -> label map
    zone_overlays = {}      # cam_id -> overlay image
    bg_gray_f = {}          # cam_id -> background float32
    last_enqueue_time = {}  # cam_id -> last inference time
    last_idle_update = {}   # cam_id -> last idle update time

    for cam_id in cam_threads:
        bg_gray_f[cam_id] = None
        last_enqueue_time[cam_id] = 0.0
        last_idle_update[cam_id] = 0.0

    # -----------------------------
    # Motion detection settings
    # -----------------------------
    MOTION_BG_ALPHA = 0.05
    MOTION_DIFF_THRESH = 15
    MOTION_RATIO_THRESH = 0.001
    MOTION_FORCE_INTERVAL = 1.0
    IDLE_REFRESH_SEC = 0.2
    MAX_DET = 10

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # -----------------------------
    # Load model
    # -----------------------------
    model = load_model()

    print("[Loop] Inference loop started.")

    # -----------------------------
    # Main loop
    # -----------------------------
    while not stop_event.is_set():
        now_loop = time.time()

        infer_cam_ids = []
        batch_frames = []

        for cam_id in list(cam_threads.keys()):
            cam = cam_threads[cam_id]
            frame = cam.read()
            if frame is None:
                continue

            # Resize frame
            small = cv2.resize(frame, resize_wh)

            # Load zones dynamically if not loaded yet
            if cam_id not in zones_scaled:
                raw_zones = get_zones_by_camera(cam_id)
                if raw_zones:
                    zs = scale_zones_from_normalized(raw_zones, small.shape)
                    zones_scaled[cam_id] = zs
                    label_maps[cam_id] = build_label_map(zs, small.shape)
                    zone_overlays[cam_id] = build_zone_overlay(zs, small.shape)
                    print(f"[Loop] Loaded {len(zs)} zone(s) for {cam_id}")

            # Motion detection
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if bg_gray_f[cam_id] is None:
                bg_gray_f[cam_id] = init_background(gray)
                motion = True
            else:
                bg_gray_f[cam_id] = update_background(
                    bg_gray_f[cam_id], gray, MOTION_BG_ALPHA
                )
                bg_u8 = bg_gray_f[cam_id].astype(np.uint8)
                motion_ratio = compute_motion_ratio(
                    gray, bg_u8,
                    diff_thresh=MOTION_DIFF_THRESH,
                    dilate_kernel=kernel,
                    dilate_iter=1
                )
                motion = motion_ratio > MOTION_RATIO_THRESH

            force = (now_loop - last_enqueue_time[cam_id]) > MOTION_FORCE_INTERVAL

            if motion or force:
                infer_cam_ids.append(cam_id)
                batch_frames.append(small)
                last_enqueue_time[cam_id] = now_loop
            else:
                # Idle refresh with zone overlay only
                if cam_id in zone_overlays and \
                        (now_loop - last_idle_update[cam_id]) >= IDLE_REFRESH_SEC:
                    idle = cv2.addWeighted(
                        small, 1.0,
                        zone_overlays[cam_id], 1.0, 0.0
                    )
                    idle = draw_camera_label(idle, cam_names.get(cam_id, cam_id))
                    with inference_frames_lock:
                        inference_frames[cam_id] = idle
                    last_idle_update[cam_id] = now_loop

        if not batch_frames:
            time.sleep(0.01)
            continue

        # -----------------------------
        # Batched inference
        # -----------------------------
        try:
            results = model.predict(
                batch_frames,
                device=DEVICE,
                classes=[0],
                conf=CONF_THRESH,
                iou=0.7,
                max_det=MAX_DET,
                verbose=False,
                imgsz=imgsz
            )
        except Exception as e:
            print(f"[Loop] Prediction error: {e}")
            continue

        # -----------------------------
        # Annotate and store frames
        # -----------------------------
        for cam_id, frame_small, result in zip(infer_cam_ids, batch_frames, results):
            annotated, intrusions = annotate_frame(
                frame=frame_small,
                result=result,
                label_map=label_maps.get(cam_id),
                zones_scaled=zones_scaled.get(cam_id),
                zone_overlay=zone_overlays.get(cam_id)
            )

            # Draw camera name on frame
            annotated = draw_camera_label(annotated, cam_names.get(cam_id, cam_id))

            # Log intrusions
            if intrusions:
                for intrusion in intrusions:
                    print(
                        f"[Loop] INTRUSION detected: "
                        f"cam={cam_id} zone={intrusion['zone']}"
                    )

            # Store annotated frame
            with inference_frames_lock:
                inference_frames[cam_id] = annotated

    # -----------------------------
    # Cleanup
    # -----------------------------
    print("[Loop] Stopping camera threads...")
    for cam_id, cam in cam_threads.items():
        try:
            cam.stop()
        except Exception:
            pass
    cam_threads.clear()

    with inference_frames_lock:
        inference_frames.clear()

    print("[Loop] Inference loop stopped.")