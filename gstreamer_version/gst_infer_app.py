import cv2
import time
import threading
import math
import numpy as np
import psutil
from datetime import datetime
from ultralytics import YOLO
import torch

from configs import *
from up_inference_core import (
    load_zones, scale_zones,
    build_label_map, build_zone_overlay,
    save_intrusion_snapshot,
)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

try:
    from alerts import send_telegram_alert
except Exception:
    def send_telegram_alert(message, snapshot_path=None):
        pass

# ------------------------------------------------------------------
# Shared state (read by main_gst.py)
# ------------------------------------------------------------------
latest_frames = {}
frame_lock = threading.Lock()
latest_seq = {i: 0 for i in range(len(CAMERA_URLS))}


# ------------------------------------------------------------------
# Camera capture (OpenCV + GStreamer string for decoding only)
# ------------------------------------------------------------------
class CameraThread:
    def __init__(self, src, cam_id=0):
        self.cam_id = cam_id
        self.src = src
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.reconnect_delay = 5
        self.pipeline = None
        self.appsink = None
        
        self._start_pipeline()
        
        if self.pipeline is not None:
            threading.Thread(target=self.update, daemon=True).start()
            print(f"[Cam {cam_id}] Stream initialized successfully.")
        else:
            print(f"[Cam {cam_id}] Failed to initialize stream.")

    def _start_pipeline(self):
        """Build and play the RTSP capture pipeline.
        Uses the exact same proven chain as grid_streamer.py."""
        pipeline_str = (
            f'rtspsrc location="{self.src}" latency=200 ! '
            f'rtph264depay ! h264parse ! avdec_h264 ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink name=sink max-buffers=2 drop=true emit-signals=true'
        )
        try:
            self.pipeline = Gst.parse_launch(pipeline_str)
            self.appsink = self.pipeline.get_by_name("sink")
            self.pipeline.set_state(Gst.State.PLAYING)
            state_ret, state, pending = self.pipeline.get_state(5 * Gst.SECOND)
            if state_ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Pipeline state change failed")
        except Exception as e:
            print(f"[Cam {self.cam_id}] ERROR: Pipeline failed to open! {e}")
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
            self.appsink = None

    def update(self):
        """Background thread: pull decoded BGR frames from appsink."""
        first_frame_received = False
        while self.running:
            if self.pipeline is None:
                time.sleep(self.reconnect_delay)
                self._start_pipeline()
                continue
            
            sample = self.appsink.emit("pull-sample")
            if sample is None:
                bus = self.pipeline.get_bus()
                msg = bus.timed_pop_filtered(0, Gst.MessageType.ERROR | Gst.MessageType.EOS)
                if msg is not None:
                    print(f"[Cam {self.cam_id}] Stream error/EOS. Reconnecting...")
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline = None
                    self.appsink = None
                time.sleep(0.001)
                continue
            
            buffer = sample.get_buffer()
            caps = sample.get_caps()
            s = caps.get_structure(0)
            w = s.get_value("width")
            h = s.get_value("height")
            
            ok, mapinfo = buffer.map(Gst.MapFlags.READ)
            if not ok:
                continue
            
            frame = None
            try:
                frame = np.frombuffer(mapinfo.data, dtype=np.uint8).reshape((h, w, 3)).copy()
            finally:
                buffer.unmap(mapinfo)
            
            if frame is None:
                continue

            if not first_frame_received:
                print(f"[Cam {self.cam_id}] First frame captured: {frame.shape}")
                first_frame_received = True
            
            with self.lock:
                self.frame = frame

    def read(self):
        """Thread-safe frame retrieval."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
            return None
    
    def stop(self):
        self.running = False
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None
# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def round_up_to_stride(x, stride=32):
    return int(math.ceil(x / stride) * stride)


def compute_motion_ratio(gray_u8, bg_u8, diff_thresh, dilate_kernel, dilate_iter):
    diff = cv2.absdiff(gray_u8, bg_u8)
    _, fg = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
    if dilate_kernel is not None and dilate_iter > 0:
        fg = cv2.dilate(fg, dilate_kernel, iterations=dilate_iter)
    nonzero = cv2.countNonZero(fg)
    return nonzero / float(gray_u8.size)


# ------------------------------------------------------------------
# Main Inference Loop
# ------------------------------------------------------------------
def inference_loop():
    cams = [CameraThread(url, cam_id=i) for i, url in enumerate(CAMERA_URLS)]
    zones_raw = [load_zones(p) for p in ZONES_PATHS]

    zones_scaled = [None] * len(cams)
    label_maps = [None] * len(cams)
    zone_overlays = [None] * len(cams)

    base_w, base_h = RESIZED_DIM
    w_r = round_up_to_stride(base_w, 32)
    h_r = round_up_to_stride(base_h, 32)
    resize_wh = (w_r, h_r)
    imgsz = (h_r, w_r)

    MOTION_BG_ALPHA = 0.05
    MOTION_DIFF_THRESH = 15
    MOTION_RATIO_THRESH = 0.001
    MOTION_FORCE_INTERVAL = 1.0
    IDLE_REFRESH_SEC = 0.2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = kernel
    dilate_iter = 1

    bg_gray_f = [None] * len(cams)
    last_idle_update = [0.0] * len(cams)
    last_enqueue_time = [0.0] * len(cams)

    model = YOLO(MODEL_PATH)

    if torch.cuda.is_available() and "cuda" in str(DEVICE).lower():
        model.to(DEVICE)
        try:
            model.fuse()
        except Exception:
            pass
        try:
            model.model.half()
        except Exception:
            pass

    prev_time = time.time()
    frame_count = 0
    fps = 0.0
    cpu = mem = 0.0
    last_sys = 0.0

    ALERT_COOLDOWN = 30
    last_alert_time = {}
    MAX_DET = 10

    while not STOP_EVENT.is_set():
        now_loop = time.time()
        infer_cam_ids = []
        batch_frames = []

        for cam_id, cam in enumerate(cams):
            frame = cam.read()
            if frame is None:
                continue

            small = cv2.resize(frame, resize_wh)

            if zones_scaled[cam_id] is None and zones_raw[cam_id]:
                zs = scale_zones(zones_raw[cam_id], frame.shape, small.shape)
                zones_scaled[cam_id] = zs
                label_maps[cam_id] = build_label_map(zs, small.shape)
                zone_overlays[cam_id] = build_zone_overlay(zs, small.shape)

            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if bg_gray_f[cam_id] is None:
                bg_gray_f[cam_id] = gray.astype(np.float32)
                motion = True
            else:
                cv2.accumulateWeighted(gray.astype(np.float32), bg_gray_f[cam_id], MOTION_BG_ALPHA)
                bg_u8 = bg_gray_f[cam_id].astype(np.uint8)
                motion_ratio = compute_motion_ratio(gray, bg_u8, MOTION_DIFF_THRESH, dilate_kernel, dilate_iter)
                motion = motion_ratio > MOTION_RATIO_THRESH

            force = (now_loop - last_enqueue_time[cam_id]) > MOTION_FORCE_INTERVAL

            if motion or force:
                infer_cam_ids.append(cam_id)
                batch_frames.append(small)
                last_enqueue_time[cam_id] = now_loop
            else:
                if zone_overlays[cam_id] is not None and (now_loop - last_idle_update[cam_id]) >= IDLE_REFRESH_SEC:
                    idle = cv2.addWeighted(small, 1.0, zone_overlays[cam_id], 1.0, 0.0)
                    with frame_lock:
                        latest_frames[cam_id] = idle
                        latest_seq[cam_id] += 1
                    last_idle_update[cam_id] = now_loop

        if not batch_frames:
            time.sleep(0.001)
            continue

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
                            message = f"🚨 INTRUSION ALERT\nCamera: {cam_id}\nZone: {intruded_zone}\nTime: {timestamp}"
                            snapshot_path = save_intrusion_snapshot(cam_id, annotated, intruded_zone)
                            threading.Thread(target=send_telegram_alert, args=(message, snapshot_path), daemon=True).start()
                    else:
                        color = (0, 255, 0)
                        label = "person"

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(annotated, (feet_x, feet_y), 5, color, -1)
                    cv2.putText(annotated, label, (x1, max(20, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            with frame_lock:
                latest_frames[cam_id] = annotated
                latest_seq[cam_id] += 1

    for cam in cams:
        cam.stop()
    print("[Inference] Loop ended.")