import cv2
import multiprocessing as mp
import time
import threading
import multiprocessing as mp
import numpy as np
import psutil
import os
import json
from datetime import datetime
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

from configs import *
from alerts import send_telegram_alert

from inference_core import CameraThread,load_zones,scale_zones,infer_worker,save_intrusion_snapshot,overlay_info


latest_frames = {}
frame_lock = threading.Lock()

# MAIN INFERENCE LOOP
# ===============================

def inference_loop():
    cams = [CameraThread(url) for url in CAMERA_URLS]
    zones_raw = [load_zones(p) for p in ZONES_PATHS]
    zones_scaled = [None] * len(cams)

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
        for cam_id, cam in enumerate(cams):
            frame = cam.read()
            if frame is None:
                continue

            small = cv2.resize(frame, RESIZED_DIM)

            if zones_scaled[cam_id] is None and zones_raw[cam_id]:
                zones_scaled[cam_id] = scale_zones(
                    zones_raw[cam_id], frame.shape, small.shape
                )

            try:
                frame_queue.put_nowait((cam_id, small))
            except:
                pass

        try:
            cam_id, frame, result = result_queue.get(timeout=0.05)
            annotated = frame.copy()

            if zones_scaled[cam_id]:
                for z in zones_scaled[cam_id]:
                    pts = np.array(z["points"], np.int32)
                    cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)
                    x0,y0 = pts[0]
                    cv2.putText(annotated,z["name"],
                                        (x0+5, y0-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
                            

            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    feet_x = int((x1+x2)/2)
                    feet_y = int(y2)
                    feet_point = Point(feet_x,feet_y)

                    intruded_zone = None
                    for z in zones_scaled[cam_id]:
                        if z["polygon"].contains(feet_point):
                            intruded_zone = z["name"]
                            break

                    if intruded_zone:
                        color = (0,0,255)
                        label =f"INTRUSION: {intruded_zone}"
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

                            # Save snapshot
                            snapshot_path = save_intrusion_snapshot(
                                cam_id, annotated, intruded_zone
                            )

                            # Telegram (threaded)
                            threading.Thread(
                                target=send_telegram_alert,
                                args=(message, snapshot_path),
                                daemon=True
                            ).start()

                            # WhatsApp (already threaded inside)
                            # send_whatsapp_alert(message)

                    else:
                        color = (0,255,0)
                        label = "person"

                    cv2.rectangle(
                        annotated,
                        (x1, y1), (x2, y2),
                        (0, 255, 0), 2
                    )
                    cv2.circle(annotated, (feet_x, feet_y), 5, color, -1)
                    cv2.putText(
                        annotated, label,
                        (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2
                    )
            if time.time() - last_sys >= 1:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                last_sys = time.time()

            annotated = overlay_info(annotated, fps, cpu, mem)

            with frame_lock:
                latest_frames[cam_id] = annotated

        except:
            pass

        frame_count += 1
        if time.time() - prev_time >= 1:
            fps = frame_count / (time.time() - prev_time)
            frame_count = 0
            prev_time = time.time()
