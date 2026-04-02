import time
import threading
import math
import cv2
import numpy as np
import psutil
from datetime import datetime
from ultralytics import YOLO

from up_inference_core import (
    CameraThread, load_zones, scale_zones,
    build_label_map, build_zone_overlay,
    save_intrusion_snapshot, overlay_info
)
from configs import *
from alerts import send_telegram_alert


latest_frames = {}
frame_lock = threading.Lock()

# NEW: per-camera update counter (used by main.py to avoid rebuilding grid unnecessarily)
latest_seq = {i: 0 for i in range(len(CAMERA_URLS))}

def round_up_to_stride(x, stride=32):
    return int(math.ceil(x / stride) * stride)


def compute_motion_ratio(gray_u8, bg_u8, diff_thresh, dilate_kernel, dilate_iter):
    diff = cv2.absdiff(gray_u8, bg_u8)
    _, fg = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    if dilate_kernel is not None and dilate_iter > 0:
        fg = cv2.dilate(fg, dilate_kernel, iterations=dilate_iter)

    nonzero = cv2.countNonZero(fg)
    return nonzero / float(gray_u8.size)


def inference_loop():
    # -----------------------------
    # Setup cameras + zones
    # -----------------------------
    cams = [CameraThread(url) for url in CAMERA_URLS]
    zones_raw = [load_zones(p) for p in ZONES_PATHS]

    zones_scaled = [None] * len(cams)   # per-cam scaled zones
    label_maps = [None] * len(cams)    # per-cam label map (H,W)->zone idx
    zone_overlays = [None] * len(cams) # per-cam static overlay

    # -----------------------------
    # Stride-safe resize + imgsz
    # -----------------------------
    # Your code uses: small = cv2.resize(frame, RESIZED_DIM)
    # OpenCV expects RESIZED_DIM = (width, height)
    base_w, base_h = RESIZED_DIM
    w_r = round_up_to_stride(base_w, 32)
    h_r = round_up_to_stride(base_h, 32)
    resize_wh = (w_r, h_r)     # (W,H) for cv2.resize
    imgsz = (h_r, w_r)        # (H,W) for Ultralytics

    # -----------------------------
    # Motion detection tuning
    # (Adjust if scene is dark / noisy)
    # -----------------------------
    MOTION_BG_ALPHA = 0.05          # background update speed
    MOTION_DIFF_THRESH = 15         # pixel diff threshold
    MOTION_RATIO_THRESH = 0.002     # motion area ratio threshold
    MOTION_FORCE_INTERVAL = 1.0     # run at least once per N seconds per camera
    IDLE_REFRESH_SEC = 0.2          # refresh grid overlay while idle

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = kernel
    dilate_iter = 1

    bg_gray_f = [None] * len(cams)       # float32 running average gray
    last_idle_update = [0.0] * len(cams)
    last_enqueue_time = [0.0] * len(cams)

    # -----------------------------
    # Single model on GPU
    # -----------------------------
    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available() and "cuda" in str(DEVICE).lower():
        idx = torch.cuda.current_device()
        print(f"[Device] Using CUDA: {torch.cuda.get_device_name(idx)} (device index: {idx})")
    else:
        print(f"[Device] Using CPU")

    # Try a few speedups if CUDA
    if "cuda" in str(DEVICE).lower():
        try:
            model.to(DEVICE)
        except Exception:
            pass
        try:
            model.fuse()
        except Exception:
            pass
        try:
            model.model.half()
        except Exception:
            pass

    # -----------------------------
    # Main loop stats + alerts
    # -----------------------------
    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    cpu = mem = 0.0
    last_sys = 0.0

    ALERT_COOLDOWN = 30
    last_alert_time = {}

    # NEW: limit detections to reduce CPU work/drawing cost
    MAX_DET = 10

    while True:
        now_loop = time.time()

        # 1) Collect frames that should be inferred
        infer_cam_ids = []
        batch_frames = []

        # Also allow idle overlay refresh for cams that are inactive
        for cam_id, cam in enumerate(cams):
            frame = cam.read()
            if frame is None:
                continue

            # Resize to stride-friendly size
            small = cv2.resize(frame, resize_wh)

            # Build zones/label_map/overlay once per camera
            if zones_scaled[cam_id] is None and zones_raw[cam_id]:
                zs = scale_zones(zones_raw[cam_id], frame.shape, small.shape)
                zones_scaled[cam_id] = zs
                label_maps[cam_id] = build_label_map(zs, small.shape)
                zone_overlays[cam_id] = build_zone_overlay(zs, small.shape)

            # Motion detection on resized frame
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if bg_gray_f[cam_id] is None:
                bg_gray_f[cam_id] = gray.astype(np.float32)
                motion = True
            else:
                cv2.accumulateWeighted(gray.astype(np.float32), bg_gray_f[cam_id], MOTION_BG_ALPHA)
                bg_u8 = bg_gray_f[cam_id].astype(np.uint8)

                motion_ratio = compute_motion_ratio(
                    gray, bg_u8,
                    diff_thresh=MOTION_DIFF_THRESH,
                    dilate_kernel=dilate_kernel,
                    dilate_iter=dilate_iter
                )
                motion = motion_ratio > MOTION_RATIO_THRESH

            force = (now_loop - last_enqueue_time[cam_id]) > MOTION_FORCE_INTERVAL

            if motion or force:
                infer_cam_ids.append(cam_id)
                batch_frames.append(small)
                last_enqueue_time[cam_id] = now_loop
            else:
                # Idle refresh: update with just zone overlay (fast)
                if zone_overlays[cam_id] is not None and (now_loop - last_idle_update[cam_id]) >= IDLE_REFRESH_SEC:
                    idle = cv2.addWeighted(small, 1.0, zone_overlays[cam_id], 1.0, 0.0)
                    with frame_lock:
                        latest_frames[cam_id] = idle
                        latest_seq[cam_id] += 1  # NEW: count idle updates
                    last_idle_update[cam_id] = now_loop

        if not batch_frames:
            continue

        # 2) Batched inference (key for smoothness)
        t0 = time.time()
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

        print("batch:", len(batch_frames), "yolo_ms:", (time.time()-t0)*1000)


        # 3) Annotate and publish
        frame_count += len(batch_frames)
        if time.time() - prev_time >= 1:
            fps = frame_count / (time.time() - prev_time)
            frame_count = 0
            prev_time = time.time()

        if time.time() - last_sys >= 1:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            last_sys = time.time()

        for cam_id, frame_small, result in zip(infer_cam_ids, batch_frames, results):
            annotated = frame_small

            if zone_overlays[cam_id] is not None:
                annotated = cv2.addWeighted(frame_small, 1.0, zone_overlays[cam_id], 1.0, 0.0)

            lm = label_maps[cam_id]
            zs = zones_scaled[cam_id]
            h, w = annotated.shape[:2]

            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # GPU->CPU sync

                for box in boxes_xyxy:
                    x1, y1, x2, y2 = map(int, box)
                    feet_x = int((x1 + x2) / 2)
                    feet_y = int(y2)

                    intruded_zone = None
                    if lm is not None and zs is not None:
                        if 0 <= feet_x < w and 0 <= feet_y < h:
                            zidx = int(lm[feet_y, feet_x])
                            if zidx != -1:
                                intruded_zone = zs[zidx]["name"]

                    if intruded_zone is None:
                        continue
                    color = (0, 0, 255)
                    label = f"INTRUSION: {intruded_zone}"
                    key = (cam_id, intruded_zone)

                    tnow = time.time()
                    if tnow - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                        last_alert_time[key] = tnow
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        message = (
                            f"🚨 INTRUSION ALERT\n"
                            f"Camera: {cam_id}\n"
                            f"Zone: {intruded_zone}\n"
                            f"Time: {timestamp}"
                        )

                        snapshot_path = save_intrusion_snapshot(cam_id, annotated, intruded_zone)
                        threading.Thread(
                            target=send_telegram_alert,
                            args=(message, snapshot_path),
                            daemon=True
                        ).start()
                    else:
                        color = (0, 255, 0)
                        label = "person"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, (feet_x, feet_y), 5, color, -1)
                    cv2.putText(
                        annotated, label,
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )

            annotated = overlay_info(annotated, fps, cpu, mem)

            with frame_lock:
                latest_frames[cam_id] = annotated
                latest_seq[cam_id] += 1  # NEW: count inference updates too