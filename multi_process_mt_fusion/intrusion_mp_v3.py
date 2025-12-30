#!/usr/bin/env python3
import cv2
import time
import json
import threading
import multiprocessing as mp
import numpy as np
import psutil
import os
from ultralytics import YOLO
from shapely.geometry import Point, Polygon

# ================= CONFIG =================
CAMERA_URLS = [
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/101",
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501"
]

ZONES_PATH1 = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam1_zones.json"
ZONES_PATH2 = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam2_zones.json"

MODEL_PATH = "yolov8n.pt"
IMG_SIZE = (416, 416)
PERSON_CLASS_ID = 0
CONF_THRES = 0.4

# =========================================

STOP_EVENT = mp.Event()

# ---------- Load Zones ----------
def load_zones(zpath):
    if not os.path.exists(zpath):
        return []
    with open(zpath, "r") as f:
        data = json.load(f)

    zones = []
    raw = data if isinstance(data, list) else data.get("zones", [])
    for z in raw:
        pts = z.get("points") or z.get("poly")
        if not pts:
            continue
        poly = Polygon(pts)
        zones.append({
            "name": z.get("name", "zone"),
            "points": pts,
            "polygon": poly
        })
    return zones


# ---------- Camera Thread ----------
class CameraThread:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running and not STOP_EVENT.is_set():
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
            else:
                if STOP_EVENT.is_set():
                    break
                print(f"[WARN] Camera dropped. Reconnecting...")
            self.cap.release()
            time.sleep(1)
            self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()


# ---------- Inference Worker ----------
def infer_worker(frame_q, result_q):
    model = YOLO(MODEL_PATH)
    model.fuse()
    model.to("cpu")

    while not STOP_EVENT.is_set():
        item = frame_q.get()
        if item is None:
            break

        cam_id, frame = item
        resized = cv2.resize(frame, IMG_SIZE)

        results = model.predict(
            resized,
            conf=CONF_THRES,
            classes=[PERSON_CLASS_ID],
            verbose=False,
            device="cpu"
        )

        result_q.put((cam_id, resized, results[0]))


# ---------- Zone Check ----------
def check_intrusion(box, zones, w, h):
    x1, y1, x2, y2 = box
    foot = Point((x1 + x2) // 2, y2)

    for z in zones:
        pts = z["points"]
        poly = Polygon([
            (int(px * w), int(py * h)) if max(map(max, pts)) <= 1 else tuple(p)
            for p in pts
        ])
        if poly.contains(foot):
            return z["name"]
    return None


# ---------- Main ----------
def main():
    zones_cam1 = load_zones(ZONES_PATH1)
    zones_cam2 = load_zones(ZONES_PATH2)
    ZONE_MAP = {0: zones_cam1, 1: zones_cam2}

    cams = [CameraThread(url) for url in CAMERA_URLS]

    ctx = mp.get_context("spawn")
    frame_q = ctx.Queue(maxsize=2)
    result_q = ctx.Queue()

    proc = ctx.Process(target=infer_worker, args=(frame_q, result_q))
    proc.start()

    fps, frame_count = 0, 0
    last_stat_time = time.time()
    cpu, mem = 0, 0

    try:
        while True:
            for i, cam in enumerate(cams):
                frame = cam.read()
                if frame is not None:
                    try:
                        frame_q.put_nowait((i, frame))
                    except:
                        pass

            try:
                cam_id, frame, result = result_q.get(timeout=0.05)
                h, w = frame.shape[:2]
                zones = ZONE_MAP.get(cam_id, [])

                if result.boxes is not None:
                    for box in result.boxes.xyxy.cpu().numpy():
                        x1, y1, x2, y2 = map(int, box)
                        intruded = check_intrusion(box, zones, w, h)

                        if intruded:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
                            cv2.putText(frame, "INTRUSION", (x1, y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                        else:
                            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 1)

                if time.time() - last_stat_time >= 1:
                    cpu = psutil.cpu_percent()
                    mem = psutil.virtual_memory().percent
                    fps = frame_count
                    frame_count = 0
                    last_stat_time = time.time()

                cv2.putText(frame, f"FPS:{fps} CPU:{cpu}% MEM:{mem}%",
                            (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                cv2.imshow(f"Camera {cam_id}", frame)

            except:
                pass

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        STOP_EVENT.set()

        for cam in cams:
            cam.stop()

        frame_q.put(None)
        proc.join(timeout=3)

        cv2.destroyAllWindows()
        print("[INFO] Clean shutdown completed")


if __name__ == "__main__":
    main()
