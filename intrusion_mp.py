import cv2
import time
import threading
import multiprocessing as mp
from ultralytics import YOLO
import numpy as np
import psutil
import os

# ---------- Threaded Camera Class for RTSP ----------
class CameraThread:
    def __init__(self, src, width=640, height=480):
        # Using CAP_FFMPEG for RTSP streams is more stable on Linux/Windows
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        # self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        
        # Note: Some RTSP streams don't allow setting width/height via OpenCV
        # We will resize manually in the worker to ensure consistency
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.src_name = str(src)[-10:] # Last 10 chars for ID
        
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        print(f"[INFO] Started Thread for: {src}")
        
    def update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            else:
                # If stream drops, try to reconnect
                print(f"[WARN] Stream {self.src_name} dropped. Reconnecting...")
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.src_name, cv2.CAP_FFMPEG)
            time.sleep(0.01) # Avoid 100% CPU usage on the reading thread

    def read(self):
        return self.frame.copy() if self.ret else None

    def stop(self):
        self.running = False
        self.cap.release()

# ---------- Inference Worker (Same as your code) ----------
def infer_worker(frame_queue, result_queue, model_path, img_size):
    model = YOLO(model_path)
    # If you have an NVIDIA GPU, this will be lightning fast. 
    # If not, change device='cpu' and remove half=True
    device = 'cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
    
    while True:
        item = frame_queue.get()
        if item is None: break
        
        cam_id, frame = item
        # Pre-process: Resize to the dimensions the model expects
        input_frame = cv2.resize(frame, img_size)
        
        # Inference
        results = model.predict(input_frame, verbose=False, device=device)
        result_queue.put((cam_id, input_frame, results[0]))

# ---------- Overlay FPS and Usage ----------
def overlay_info(frame, fps):
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    text = f"FPS: {fps:.1f} | CPU: {cpu}% | MEM: {mem}%"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# ---------- Main Execution ----------
def main():
    # 1. ADD YOUR RTSP URLS HERE
    camera_urls = [
        "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/101",
        "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501"
    ]
    
    model_path = 'yolov8n.pt'
    resized_dim = (640, 480) 

    # 2. Initialize Camera Threads
    cams = [CameraThread(url) for url in camera_urls]
    
    # Context setup for Multiprocessing
    ctx = mp.get_context('spawn') 
    frame_queue = ctx.Queue(maxsize=4) # Limit queue size to prevent memory bloat
    result_queue = ctx.Queue()

    # 3. Start Inference Processes (One per camera)
    processes = []
    for _ in range(len(cams)):
        p = ctx.Process(target=infer_worker, args=(frame_queue, result_queue, model_path, resized_dim))
        p.start()
        processes.append(p)

    prev_time = time.time()
    frame_count = 0
    fps = 0

    try:
        while True:
            # PUSH: Send latest frames to queue
            for i, cam in enumerate(cams):
                frame = cam.read()
                if frame is not None:
                    # use nowait to avoid blocking if the queue is full
                    try:
                        frame_queue.put_nowait((i, frame))
                    except:
                        pass # Skip if queue is busy

            # PULL: Get processed results
            for _ in range(len(cams)):
                try:
                    # We use a timeout so the loop doesn't hang if a process is slow
                    cam_id, frame, result = result_queue.get(timeout=0.1)
                    annotated = result.plot()
                    annotated = overlay_info(annotated, fps)
                    
                    cv2.imshow(f"Camera {cam_id}", annotated)
                except:
                    continue

            # FPS Logic
            frame_count += 1
            if time.time() - prev_time >= 1.0:
                fps = frame_count / (time.time() - prev_time)
                frame_count = 0
                prev_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        # Cleanup
        for cam in cams: cam.stop()
        for _ in range(len(processes)): frame_queue.put(None)
        for p in processes: p.join()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()