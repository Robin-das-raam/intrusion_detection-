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
import torch

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# ===============================
# GLOBALS
# ===============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE}")

cv2.setUseOptimized(True)
cv2.setNumThreads(1)

CAMERA_URLS = [
    # "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501",
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/101",
]

ZONES_PATHS = [
    # "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam2_zones.json",
    "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam1_zones.json",
]

MODEL_PATH = "yolov8n.pt"
RESIZED_DIM = (640, 420)
CONF_THRESH = 0.4
QUEUE_SIZE = 4

STOP_EVENT = mp.Event()

latest_frames = {}
frame_lock = threading.Lock()

# ===============================
# CAMERA THREAD
# ===============================

class CameraThread:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.ret, self.frame = self.cap.read()
        self.running = True

        threading.Thread(target=self.update, daemon=True).start()
        print(f"[INFO] Camera thread started")

    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                if STOP_EVENT.is_set():
                    break
                time.sleep(1)

    def read(self):
        return self.frame.copy() if self.ret else None

    def stop(self):
        self.running = False
        self.cap.release()

# ===============================
# ZONE LOGIC
# ===============================

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

# ===============================
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

# ===============================
# FASTAPI
# ===============================

app = FastAPI()

def mjpeg_generator(cam_id: int):
    while True:
        with frame_lock:
            frame = latest_frames.get(cam_id)

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes() +
            b"\r\n"
        )

@app.get("/live/{cam_id}")
def live(cam_id: int):
    return StreamingResponse(
        mjpeg_generator(cam_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("startup")
def startup_event():
    threading.Thread(target=inference_loop, daemon=True).start()
