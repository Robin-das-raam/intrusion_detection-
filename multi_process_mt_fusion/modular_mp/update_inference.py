import cv2
import multiprocessing as mp
import time
import threading
import numpy as np
import psutil
import os
import json
from datetime import datetime
from queue import Empty

from up_inference_core import (
    CameraThread, load_zones, scale_zones,
    build_label_map, build_zone_overlay,
    infer_worker, save_intrusion_snapshot, overlay_info
)

from configs import *
from alerts import send_telegram_alert


latest_frames = {}
frame_lock = threading.Lock()


# MAIN INFERENCE LOOP
# ===============================

def inference_loop():
    cams = [CameraThread(url) for url in CAMERA_URLS]
    zones_raw = [load_zones(p) for p in ZONES_PATHS]

    zones_scaled = [None] * len(cams)       # list of {name, pts}
    label_maps = [None] * len(cams)       # pixel -> zone idx
    zone_overlays = [None] * len(cams)    # static polyline+text overlay

    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    result_queue = ctx.Queue()

    workers = []
    for _ in range(min(len(cams), os.cpu_count() // 2)):
        p = ctx.Process(target=infer_worker, args=(frame_queue, result_queue), daemon=True)
        p.start()
        workers.append(p)

    prev_time = time.time()
    frame_count = 0
    fps = 0
    cpu = mem = 0
    last_sys = 0
    ALERT_COOLDOWN = 30
    last_alert_time = {}

    while True:
        # 1) capture frames for all cams
        for cam_id, cam in enumerate(cams):
            frame = cam.read()
            if frame is None:
                continue

            small = cv2.resize(frame, RESIZED_DIM)

            # build scaled zones + label_map + overlay once
            if zones_scaled[cam_id] is None and zones_raw[cam_id]:
                zs = scale_zones(zones_raw[cam_id], frame.shape, small.shape)
                zones_scaled[cam_id] = zs
                label_maps[cam_id] = build_label_map(zs, small.shape)
                zone_overlays[cam_id] = build_zone_overlay(zs, small.shape)

            try:
                frame_queue.put_nowait((cam_id, small))
            except:
                pass

        # 2) drain results_queue and keep newest per camera (reduces backlog latency)
        results_by_cam = {}
        while True:
            try:
                cam_id, frame, result = result_queue.get_nowait()
                results_by_cam[cam_id] = (frame, result)  # overwrite => newest for that cam
            except Empty:
                break

        # nothing finished yet
        if not results_by_cam:
            continue

        # 3) annotate newest results
        for cam_id, (frame, result) in results_by_cam.items():
            # overlay static zone outlines (pre-rendered)
            if zone_overlays[cam_id] is not None:
                annotated = cv2.addWeighted(frame, 1.0, zone_overlays[cam_id], 1.0, 0.0)
            else:
                annotated = frame.copy()

            # update system stats (same logic as your code)
            if time.time() - last_sys >= 1:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                last_sys = time.time()

            # intrusion logic using label_map (O(1) per detection)
            lm = label_maps[cam_id]
            zs = zones_scaled[cam_id]
            h, w = frame.shape[:2]

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
                        now = time.time()

                        if now - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                            last_alert_time[key] = now
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            message = (
                                f"🚨 INTRUSION ALERT\n"
                                f"Camera: {cam_id}\n"
                                f"Zone: {intruded_zone}\n"
                                f"Time: {timestamp}"
                            )

                            snapshot_path = save_intrusion_snapshot(
                                cam_id, annotated, intruded_zone
                            )

                            threading.Thread(
                                target=send_telegram_alert,
                                args=(message, snapshot_path),
                                daemon=True
                            ).start()
                    else:
                        color = (0, 255, 0)
                        label = "person"

                    # draw detection
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, (feet_x, feet_y), 5, color, -1)
                    cv2.putText(
                        annotated,
                        label,
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
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