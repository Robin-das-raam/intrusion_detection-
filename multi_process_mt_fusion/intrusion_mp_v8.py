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
import requests
from datetime import datetime
import pywhatkit as kit
import threading

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
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501",
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/101",
]

ZONES_PATHS = [
    "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam2_zones.json",
    "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam1_zones.json",
]



MODEL_PATH = "yolov8n.pt"
RESIZED_DIM = (640, 420)
CONF_THRESH = 0.4
QUEUE_SIZE = 4

STOP_EVENT = mp.Event()


latest_frames = {}
combined_frame = None
frame_lock = threading.Lock()



## Telegram Bot Credentials
BOT_TOKEN = "8214541766:AAHFrh4efpd7VdTBPYQY5Mv0QYDYQ24_jY4"
CHAT_ID = "6813192996"

## WhatsApp Alert Config
WHATSAPP_NUMBER = ""  # replace with your number in international format


# ------------------------------
# ALERT FUNCTIONS
# ------------------------------
def send_telegram_alert(message, image_path=None):
    """Send alert message + optional image to Telegram"""
    try:
        # Send text message
        url_msg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url_msg, data={"chat_id": CHAT_ID, "text": message})

        # Send image if exists
        if image_path and os.path.exists(image_path):
            url_photo = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                payload = {"chat_id": CHAT_ID, "caption": "📸 Intrusion Snapshot"}
                requests.post(url_photo, data=payload, files={"photo": photo})
            print(f"✅ Telegram alert sent for {image_path}")

    except Exception as e:
        print(f"❌ Telegram alert failed: {e}")


def send_whatsapp_alert(message):
    """Send WhatsApp message asynchronously using pywhatkit"""
    try:
        def send_msg():
            kit.sendwhatmsg_instantly(
                phone_no=WHATSAPP_NUMBER,
                message=message,
                wait_time=10,
                tab_close=True
            )
            print("✅ WhatsApp alert sent successfully")

        # Run in a separate thread so main video loop doesn’t freeze
        threading.Thread(target=send_msg, daemon=True).start()

    except Exception as e:
        print(f"❌ WhatsApp alert failed: {e}")


# def build_grid(frames, rows, cols):
#     h, w, c = frames[0].shape
#     canvas = np.zeros((h*rows, w*cols, c), dtype=np.uint8)

#     for i, frame in enumerate(frames):
#         r = i // cols
#         c_ = i % cols
#         canvas[r*h:(r+1)*h, c_*w:(c_+1)*w] = frame

#     return canvas

def get_grid_shape(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def build_fixed_size_grid(frames, target_size):
    """
    frames: list of images already annotated (each RESIZED_DIM)
    target_size: (W, H) = RESIZED_DIM
    """
    W, H = target_size
    n = len(frames)

    rows, cols = get_grid_shape(n)

    cell_w = W // cols
    cell_h = H // rows

    grid = np.zeros((H, W, 3), dtype=np.uint8)

    for i, frame in enumerate(frames):
        r = i // cols
        c = i % cols

        resized = cv2.resize(frame, (cell_w, cell_h))

        y1 = r * cell_h
        y2 = y1 + cell_h
        x1 = c * cell_w
        x2 = x1 + cell_w

        grid[y1:y2, x1:x2] = resized

    return grid


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

def save_intrusion_snapshot(cam_id, frame, zone_name):
    os.makedirs("alert_snapshots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"alert_snapshots/cam{cam_id}_{zone_name}_{ts}.jpg"
    cv2.imwrite(path, frame)
    return path

# ===============================
# MAIN INFERENCE LOOP
# ===============================

def inference_loop():
    global combined_frame

    cams = [CameraThread(url) for url in CAMERA_URLS]
    zones_raw = [load_zones(p) for p in ZONES_PATHS]
    zones_scaled = [None] * len(cams)

    # -------------------------
    # MULTIPROCESS INFERENCE
    # -------------------------
    ctx = mp.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=QUEUE_SIZE)
    result_queue = ctx.Queue()

    workers = []
    for _ in range(min(len(cams), os.cpu_count() // 2)):
        p = ctx.Process(
            target=infer_worker,
            args=(frame_queue, result_queue),
            daemon=True
        )
        p.start()
        workers.append(p)

    # -------------------------
    # METRICS
    # -------------------------
    prev_time = time.time()
    frame_count = 0
    fps = 0
    cpu = mem = 0
    last_sys = 0

    # -------------------------
    # ALERT STATE
    # -------------------------
    ALERT_COOLDOWN = 30  # seconds
    last_alert_time = {}

    while True:
        # -------------------------
        # READ FRAMES
        # -------------------------
        for cam_id, cam in enumerate(cams):
            frame = cam.read()
            if frame is None:
                continue

            frame = cv2.resize(frame, RESIZED_DIM)

            if zones_scaled[cam_id] is None and zones_raw[cam_id]:
                zones_scaled[cam_id] = scale_zones(
                    zones_raw[cam_id],
                    frame.shape,
                    frame.shape
                )

            try:
                frame_queue.put_nowait((cam_id, frame))
            except:
                pass

        # -------------------------
        # GET INFERENCE RESULT
        # -------------------------
        try:
            cam_id, frame, result = result_queue.get(timeout=0.05)
            annotated = frame.copy()

            # Draw zones
            if zones_scaled[cam_id]:
                for z in zones_scaled[cam_id]:
                    pts = np.array(z["points"], np.int32)
                    cv2.polylines(annotated, [pts], True, (255, 0, 0), 2)
                    x0, y0 = pts[0]
                    cv2.putText(
                        annotated,
                        z["name"],
                        (x0 + 5, y0 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2
                    )

            # Process detections
            if result.boxes is not None:
                for box in result.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = map(int, box)
                    feet_x = (x1 + x2) // 2
                    feet_y = y2
                    feet_point = Point(feet_x, feet_y)

                    intruded_zone = None
                    for z in zones_scaled[cam_id]:
                        if z["polygon"].contains(feet_point):
                            intruded_zone = z["name"]
                            break

                    if intruded_zone:
                        key = (cam_id, intruded_zone)
                        now = time.time()

                        if now - last_alert_time.get(key, 0) > ALERT_COOLDOWN:
                            last_alert_time[key] = now

                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            msg = (
                                f"🚨 INTRUSION ALERT\n"
                                f"Camera: {cam_id}\n"
                                f"Zone: {intruded_zone}\n"
                                f"Time: {timestamp}"
                            )

                            snapshot = save_intrusion_snapshot(
                                cam_id, annotated, intruded_zone
                            )

                            threading.Thread(
                                target=send_telegram_alert,
                                args=(msg, snapshot),
                                daemon=True
                            ).start()

                            send_whatsapp_alert(msg)

                        color = (0, 0, 255)
                        label = f"INTRUSION: {intruded_zone}"
                    else:
                        color = (0, 255, 0)
                        label = "person"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
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

            # -------------------------
            # SYSTEM INFO
            # -------------------------
            if time.time() - last_sys >= 1:
                cpu = psutil.cpu_percent()
                mem = psutil.virtual_memory().percent
                last_sys = time.time()

            annotated = overlay_info(annotated, fps, cpu, mem)

            # -------------------------
            # UPDATE GRID VIEW
            # -------------------------
            with frame_lock:
                latest_frames[cam_id] = annotated

                if len(latest_frames) == len(cams):
                    frames = [latest_frames[i] for i in sorted(latest_frames)]
                    n = len(frames)
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    # combined_frame = build_grid(frames, rows, cols)
                    combined_frame = build_fixed_size_grid(frames,RESIZED_DIM)


                 
        except:
            pass

        # -------------------------
        # FPS
        # -------------------------
        frame_count += 1
        if time.time() - prev_time >= 1:
            fps = frame_count / (time.time() - prev_time)
            frame_count = 0
            prev_time = time.time()

# ===============================
# FASTAPI
# ===============================

app = FastAPI()

# def mjpeg_grid_generator():
#     while True:
#         with frame_lock:
#             f0 = latest_frames.get(0)
#             f1 = latest_frames.get(1)

#         if f0 is None or f1 is None:
#             time.sleep(0.03)
#             continue

#         # Resize to same height
#         h = min(f0.shape[0], f1.shape[0])
#         w = int(h * f0.shape[1] / f0.shape[0])

#         f0r = cv2.resize(f0, (w, h))
#         f1r = cv2.resize(f1, (w, h))

#         grid = np.hstack((f0r, f1r))

#         ok, jpeg = cv2.imencode(".jpg", grid)
#         if not ok:
#             continue

#         yield (
#             b"--frame\r\n"
#             b"Content-Type: image/jpeg\r\n\r\n" +
#             jpeg.tobytes() +
#             b"\r\n"
#         )

#         time.sleep(0.03)  # ~30 FPS




def mjpeg_grid_generator():
    while True:
        with frame_lock:
            frames = [latest_frames.get(i) for i in range(len(CAMERA_URLS))]

        if any(f is None for f in frames):
            time.sleep(0.05)
            continue

        # ❗ NO RESIZE HERE
        grid = np.hstack(frames)

        ret, buffer = cv2.imencode(".jpg", grid)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.get("/live/grid")
def live_grid():
    return StreamingResponse(
        mjpeg_grid_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("startup")
def startup_event():
    threading.Thread(target=inference_loop, daemon=True).start()
