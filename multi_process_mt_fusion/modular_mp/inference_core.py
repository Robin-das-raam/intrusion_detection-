import cv2
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



###### CAMERA THREAD #########

class CameraThread:
    def __init__(self,src):
        self.cap = cv2.VideoCapture(src,cv2.CAP_FFMPEG)
        self.ret, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update,daemon=True).start()

    def update(self):
        while self.running:
            ret,frame = self.cap.read()
            if ret:
                self.ret, self.frame = ret, frame

            else:
                time.sleep(1)

    def read(self):
        return self.frame.copy() if self.ret else None
    
    def stop(self):
        self.running = False
        self.cap.release()


###### ZONES ################

def load_zones(zpath):
    if not os.path.exists(zpath):
        return []

    with open(zpath, "r") as f:
        data = json.load(f)

    raw = data if isinstance(data, list) else data.get("zones", [])
    zones = []

    for z in raw:
        zones.append({
            "name": z.get("name", "zone"),
            "points": z.get("points", [])
        })

    return zones

def scale_zones(zones, src_shape, dst_shape):
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]

    scaled = []
    for z in zones:
        pts = []
        for x, y in z["points"]:
            nx = x / src_w
            ny = y / src_h
            pts.append((int(nx * dst_w), int(ny * dst_h)))

        scaled.append({
            "name": z["name"],
            "points": pts,
            "polygon": Polygon(pts)
        })

    return scaled

# ===============================
# INFERENCE WORKER
# ===============================
def infer_worker(frame_queue, result_queue):
    model = YOLO(MODEL_PATH).to(DEVICE)

    while True:
        item = frame_queue.get()
        if item is None:
            break

        cam_id, frame = item

        results = model.predict(
            frame,
            device=DEVICE,
            classes=[0],
            conf=CONF_THRESH,
            verbose=False
        )

        result_queue.put((cam_id, frame, results[0]))

# ===============================
# OVERLAY
# ===============================

def overlay_info(frame, fps, cpu, mem):
    text = f"FPS:{fps:.1f} | CPU:{cpu}% | MEM:{mem}%"
    cv2.putText(frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def save_intrusion_snapshot(cam_id, frame, zone_name):
    os.makedirs("alert_snapshots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"alert_snapshots/cam{cam_id}_{zone_name}_{ts}.jpg"
    cv2.imwrite(path, frame)
    return path
