import time
import threading
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from turbojpeg import TurboJPEG

from configs import CAMERA_URLS

app = FastAPI()
jpeg = TurboJPEG()

grid_buffer = [None, None]
current_grid_idx = 0
grid_lock = threading.Lock()

N_CAM = len(CAMERA_URLS)

# Cache last frames per camera so grid never stalls
last_cam_frames = [None] * N_CAM
last_lock = threading.Lock()  # protects last_cam_frames

def build_grid(frames, cols=2):
    if not frames:
        return None
    h, w, _ = frames[0].shape
    rows = (len(frames) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame
    return grid

def update_grid_from_cached():
    global grid_buffer, current_grid_idx
    frames = last_cam_frames
    if any(f is None for f in frames):
        return

    new_grid = build_grid(frames, cols=2)

    with grid_lock:
        current_grid_idx = 1 - current_grid_idx
        grid_buffer[current_grid_idx] = new_grid

@app.get("/live_grid")
def live_grid():
    def generator():
        frame_count = 0
        start_time = time.time()

        while True:
            with grid_lock:
                grid = grid_buffer[current_grid_idx]

            if grid is None:
                time.sleep(0.005)
                continue

            jpeg_bytes = jpeg.encode(grid, quality=80)
            frame_count += 1

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[Script1] Stream FPS: {fps:.2f}")

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                jpeg_bytes +
                b"\r\n"
            )

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("startup")
def start():
    from update_inference_1 import inference_loop, latest_frames, frame_lock

    threading.Thread(target=inference_loop, daemon=True).start()

    def grid_updater():
        # Fixed rate publisher (smooth output), uses cached frames
        while True:
            with frame_lock:
                for i in range(N_CAM):
                    f = latest_frames.get(i)
                    if f is not None:
                        last_cam_frames[i] = f

            update_grid_from_cached()
            time.sleep(0.03)  # ~33 fps grid publishes

    threading.Thread(target=grid_updater, daemon=True).start()