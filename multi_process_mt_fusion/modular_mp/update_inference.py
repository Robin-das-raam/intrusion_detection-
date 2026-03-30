import cv2
import multiprocessing as mp
import time
import threading
import numpy as np
import psutil
import os
from queue import Empty
from datetime import datetime

from up_inference_core import (
    CameraThread, load_zones, scale_zones,
    build_label_map, build_zone_overlay,
    infer_worker, save_intrusion_snapshot, overlay_info
)
from configs import *
from alerts import send_telegram_alert

latest_frames = {}
frame_lock = threading.Lock()


# -----------------------------
# Motion detection helper
# -----------------------------
def compute_motion_ratio(gray_u8, bg_u8, diff_thresh, dilate_kernel, dilate_iter):
    """
    gray_u8: (H,W) uint8 current gray frame
    bg_u8:   (H,W) uint8 background model (slowly updated)
    returns: motion_ratio in [0..1]
    """
    diff = cv2.absdiff(gray_u8, bg_u8)
    _, fg = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)

    if dilate_kernel is not None and dilate_iter > 0:
        fg = cv2.dilate(fg, dilate_kernel, iterations=dilate_iter)

    nonzero = cv2.countNonZero(fg)
    return nonzero / float(gray_u8.size)


def inference_loop():
    cams = [CameraThread(url) for url in CAMERA_URLS]
    zones_raw = [load_zones(p) for p in ZONES_PATHS]

    zones_scaled = [None] * len(cams)     # {name, pts, points}
    label_maps = [None] * len(cams)     # pixel -> zone idx
    zone_overlays = [None] * len(cams)  # static overlay (polylines + zone names)

    # -----------------------------
    # Motion detection state
    # -----------------------------
    # Tune these in configs if you want:
    MOTION_BG_ALPHA = 0.02          # how quickly background adapts (0.01-0.05 typical)
    MOTION_DIFF_THRESH = 25        # pixel intensity difference threshold
    MOTION_RATIO_THRESH = 0.01     # ratio threshold to consider as motion

    # Optional: hysteresis/forcing
    MOTION_FORCE_INTERVAL = 3.0    # even if no motion, run YOLO at least every N seconds per camera
    IDLE_REFRESH_SEC = 0.25         # when no motion, update grid with cheap frame at most this often

    # Background model per camera (float32 running average)
    bg_gray_f = [None] * len(cams)
    last_idle_update = [0.0] * len(cams)
    last_enqueue_time = [0.0] * len(cams)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = kernel
    dilate_iter = 1

    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    result_queue = ctx.Queue()

    workers = []
    # Usually best to start with 1 worker if you're on a single GPU
    # Increase if you know GPU can handle it.
    
    for _ in range(min(len(cams), os.cpu_count() // 2)):
        p = ctx.Process(target=infer_worker, args=(frame_queue, result_queue), daemon=True)
        p.start()
        workers.append(p)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    cpu = mem = 0.0
    last_sys = 0.0

    ALERT_COOLDOWN = 30
    last_alert_time = {}

    while True:
        now_loop = time.time()

        # 1) Read + resize + motion gating + enqueue
        for cam_id, cam in enumerate(cams):
            frame = cam.read()
            if frame is None:
                continue

            small = cv2.resize(frame, RESIZED_DIM)

            # Build zone scaling + overlay ONCE per camera
            if zones_scaled[cam_id] is None and zones_raw[cam_id]:
                zs = scale_zones(zones_raw[cam_id], frame.shape, small.shape)
                zones_scaled[cam_id] = zs
                label_maps[cam_id] = build_label_map(zs, small.shape)
                zone_overlays[cam_id] = build_zone_overlay(zs, small.shape)

            # ----- Motion detection (on resized frame) -----
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if bg_gray_f[cam_id] is None:
                # init background model
                bg_gray_f[cam_id] = gray.astype(np.float32)
                motion = True  # first time, allow inference
                motion_ratio = 1.0
            else:
                # Update background model (slowly)
                cv2.accumulateWeighted(gray.astype(np.float32), bg_gray_f[cam_id], MOTION_BG_ALPHA)

                bg_u8 = bg_gray_f[cam_id].astype(np.uint8)
                motion_ratio = compute_motion_ratio(
                    gray, bg_u8,
                    diff_thresh=MOTION_DIFF_THRESH,
                    dilate_kernel=dilate_kernel,
                    dilate_iter=dilate_iter
                )
                motion = motion_ratio > MOTION_RATIO_THRESH

            # Decide enqueue vs idle refresh
            force = (now_loop - last_enqueue_time[cam_id]) > MOTION_FORCE_INTERVAL

            if motion or force:
                try:
                    frame_queue.put_nowait((cam_id, small))
                    last_enqueue_time[cam_id] = now_loop
                except:
                    pass
            else:
                # Cheap idle update (zone overlay only) to keep grid responsive
                if zone_overlays[cam_id] is not None and (now_loop - last_idle_update[cam_id]) >= IDLE_REFRESH_SEC:
                    idle = cv2.addWeighted(small, 1.0, zone_overlays[cam_id], 1.0, 0.0)
                    with frame_lock:
                        latest_frames[cam_id] = idle
                    last_idle_update[cam_id] = now_loop

        # 2) Drain results_queue (keep newest per camera)
        results_by_cam = {}
        while True:
            try:
                cam_id, frame_small, result = result_queue.get_nowait()
                results_by_cam[cam_id] = (frame_small, result)  # overwrite => newest
            except Empty:
                break

        if not results_by_cam:
            # update fps counter even if no results
            if time.time() - prev_time >= 1:
                fps = frame_count / (time.time() - prev_time)
                frame_count = 0
                prev_time = time.time()
            continue

        # 3) Annotate newest results
        for cam_id, (frame_small, result) in results_by_cam.items():
            annotated = frame_small

            # overlay static zone outlines (cheap)
            if zone_overlays[cam_id] is not None:
                annotated = cv2.addWeighted(frame_small, 1.0, zone_overlays[cam_id], 1.0, 0.0)

            # system stats
            if time.time() - last_sys >= 1:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                last_sys = time.time()

            # intrusion logic using label map (O(1) point->zone lookup)
            lm = label_maps[cam_id]
            zs = zones_scaled[cam_id]
            h, w = annotated.shape[:2]

            if result.boxes is not None:
                boxes_xyxy = result.boxes.xyxy.cpu().numpy()

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

                    if intruded_zone:
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

        # fps calc
        frame_count += 1
        if time.time() - prev_time >= 1:
            fps = frame_count / (time.time() - prev_time)
            frame_count = 0
            prev_time = time.time()