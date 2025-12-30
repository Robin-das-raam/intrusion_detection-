import cv2
import time
import threading
import multiprocessing as mp
from ultralytics import YOLO
import numpy as np
import psutil
import os
import json
from shapely.geometry import Point, Polygon

#### OPENCV CPU tuning
cv2.setUseOptimized(True)
cv2.setNumThreads(1)

## Config

CAMERA_URLS = [
        # "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501",
        "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/101"
    ]

ZONES_PATHS = [
    "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam2_zones.json",
    # "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam1_zones.json",
]

MODEL_PATH = "yolov8n.pt"
RESIZED_DIM = (416, 416)        # Optimized resolution
CONF_THRESH = 0.4               # Confidence threshold
QUEUE_SIZE = 4


STOP_EVENT = mp.Event()

# Threaded Camera Reader

class CameraThread:
    def __init__(self,src):
        self.src = src
        self.cap = cv2.VideoCapture(src,cv2.CAP_FFMPEG)
        self.ret,self.frame = self.cap.read()
        self.running = True

        t = threading.Thread(target=self.update, daemon = True)
        t.start()
        print(f"[INFO] Camera thread started")


    def update(self):
        while self.running:
            ret,frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

            else:
                if STOP_EVENT.is_set():
                    break
                # print("[WARN] Camera dropped. Reconecting...")
                # self.cap.release()
                # time.sleep(2)
                # self.cap = cv2.VideoCapture(self.src,cv2.CAP_FFMPEG)

                time.sleep(1)

    def read(self):
        return self.frame.copy() if self.ret else None
    
    def stop(self):
        self.running = False
        self.cap.release()


#Load Zone
# ---- load zone files (expects list of zones: [{"name":..., "points":[[x,y],...]}]) ----
# def load_zones(zpath):
#     if not os.path.exists(zpath):
#         print(f"[zone] not found: {zpath} -> using empty list")
#         return []
#     with open(zpath, "r") as f:
#         data = json.load(f)
#     zones = []
#     # Accept either a list at top level or {"zones": [...]}
#     raw = data if isinstance(data, list) else data.get("zones", [])
#     for z in raw:
#         name = z.get("name") or z.get("id") or f"zone_{len(zones)+1}"
#         pts = z.get("points") or z.get("poly") or []
#         if not pts:
#             print(f"[zone] warning: zone {name} has no points, skipping")
#             continue
#         try:
#             poly = Polygon(pts)
#             zones.append({"name": name, "polygon": poly, "points": pts})
#         except Exception as e:
#             print(f"[zone] invalid polygon for {name}: {e}")
#     return zones

from shapely.geometry import Polygon
import os, json

def load_zones(zpath):
    """
    Loads zones and converts them to NORMALIZED coordinates (0–1).
    Works for any inference resolution.
    """
    if not os.path.exists(zpath):
        print(f"[zone] not found: {zpath} -> using empty list")
        return []

    with open(zpath, "r") as f:
        data = json.load(f)

    zones = []
    raw = data if isinstance(data, list) else data.get("zones", [])

    for z in raw:
        name = z.get("name") or z.get("id") or f"zone_{len(zones)+1}"
        pts = z.get("points") or z.get("poly") or []

        if not pts:
            print(f"[zone] warning: zone {name} has no points, skipping")
            continue

        pts = [(float(x), float(y)) for x, y in pts]

        # If values look like pixels → normalize later using frame size
        zones.append({
            "name": name,
            "points": pts,   # still raw here
            "normalized": False
        })

    return zones


## Inference Worker(CPU)

def infer_worker(frame_queue, result_queue, model_path):
    model = YOLO(model_path)
    model.fuse()   #CPU optimization

    while True:
        item = frame_queue.get()
        if item is None:
            break

        cam_id, frame = item

        results = model.predict(
            frame,
            device = "cpu",
            classes = [0],
            conf = CONF_THRESH,
            verbose = False
        )

        result_queue.put((cam_id,frame, results[0]))


## OVERLAY FPS/CPU/RAM

def overlay_info(frame,fps,cpu,mem):
    text = f"FPS:{fps:.1f} | CPU:{cpu}% | MEM:{mem}%"
    cv2.putText(
        frame,text,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,
        (0,255,0),2
    )
    return frame

### Main

def main():
    # Camera threads
    cams = [CameraThread(url) for url in CAMERA_URLS]

    # load zones per camera
    zones_per_cam = [load_zones(p) for p in ZONES_PATHS]

    # Multiprocessing context
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    result_queue = ctx.Queue()

    # Reduce inference workers (CPU-aware)
    NUM_INFER_WORKERS = min(len(cams), max(1, os.cpu_count() // 2))
    print(f"[INFO] Starting {NUM_INFER_WORKERS} inference workers")

    processes = []
    for _ in range(NUM_INFER_WORKERS):
        p = ctx.Process(
            target=infer_worker,
            args=(frame_queue, result_queue, MODEL_PATH),
            daemon=True
        )
        p.start()
        processes.append(p)

    # FPS + system stats
    prev_time = time.time()
    frame_count = 0
    fps = 0

    last_sys_time = 0
    cpu = mem = 0

    try:
        while True:
            # -------------------------------
            # PUSH frames → inference
            # -------------------------------
            for cam_id, cam in enumerate(cams):
                frame = cam.read()
                if frame is None:
                    continue

                small = cv2.resize(frame, RESIZED_DIM)

                try:
                    frame_queue.put_nowait((cam_id, small))
                except:
                    pass  # drop frame if queue is full

            # -------------------------------
            # PULL results ← inference
            # -------------------------------

            for _ in range(len(cams)):
                try:
                    cam_id, frame, result = result_queue.get(timeout=0.05)

                    annotated = frame.copy()

                    # Draw zones
                    for z in zones_per_cam[cam_id]:
                        pts = np.array(z["points"], dtype=np.int32)
                        cv2.polylines(annotated,[pts],True,(255,0,0),2)
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
                            for z in zones_per_cam[cam_id]:
                                if z["polygon"].contains(feet_point):
                                    intruded_zone = z["name"]
                                    break

                            if intruded_zone:
                                color = (0,0,255)
                                label =f"INTRUSION: {intruded_zone}"

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

                    # Update CPU/RAM once per second
                    if time.time() - last_sys_time >= 1:
                        cpu = psutil.cpu_percent()
                        mem = psutil.virtual_memory().percent
                        last_sys_time = time.time()

                    annotated = overlay_info(annotated, fps, cpu, mem)
                    cv2.imshow(f"Camera {cam_id}", annotated)

                except:
                    pass

            # -------------------------------
            # FPS calculation
            # -------------------------------
            frame_count += 1
            if time.time() - prev_time >= 1:
                fps = frame_count / (time.time() - prev_time)
                frame_count = 0
                prev_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("[INFO] Shutting down...")
        STOP_EVENT.set()
        for cam in cams:
            cam.stop()

        for _ in processes:
            frame_queue.put(None)

        for p in processes:
            p.join()

        cv2.destroyAllWindows()

# ===============================
# ENTRY POINT
# ===============================
if __name__ == "__main__":
    main()
