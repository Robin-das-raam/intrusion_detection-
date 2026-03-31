import time
import threading
import numpy as np
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from turbojpeg import TurboJPEG  

app = FastAPI()
jpeg = TurboJPEG()  # Faster JPEG encoding

# Double buffering: Two grids to avoid locking
grid_buffer = [None, None]
current_grid_idx = 0
grid_lock = threading.Lock()



def build_grid(frames, cols=2):
    """Optimized grid construction (pre-allocated)."""
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

def update_grid(frames):
    """Update grid_buffer without blocking inference."""
    global grid_buffer, current_grid_idx

    new_grid = build_grid(frames, cols=2)
    if new_grid is None:
        return

    with grid_lock:
        current_grid_idx = 1 - current_grid_idx  # Toggle buffer
        grid_buffer[current_grid_idx] = new_grid

@app.get("/live_grid")
def live_grid():
    def generator():
        last_grid_idx = -1
        frame_count = 0
        start_time = time.time()
        while True:
            ##t0
            t0 = time.time()

            with grid_lock:
                if grid_buffer[current_grid_idx] is None:
                    time.sleep(0.01)
                    continue
                if current_grid_idx == last_grid_idx:
                    time.sleep(0.01)  # Skip if no new grid
                    continue
                last_grid_idx = current_grid_idx
                grid = grid_buffer[current_grid_idx]
            ##t1
            t1 = time.time()

            # Encode JPEG (TurboJPEG is 3-5x faster than OpenCV)
            jpeg_bytes = jpeg.encode(grid, quality=80)

            t2 = time.time()

            frame_count +=1

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[Script1] FPS: {fps:.2f}, Encode Time: {(t2-t1)*1000:.2f} ms")


            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes +
                b"\r\n"
            )

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("startup")
def start():
    global latest_frames, frame_lock

    from update_inference_1 import inference_loop, latest_frames, frame_lock

    # Start inference thread
    threading.Thread(target=inference_loop, daemon=True).start()

    # Start grid updater thread
    def grid_updater():
        while True:
            with frame_lock:
                frames = list(latest_frames.values())
            update_grid(frames)
            time.sleep(0.03)  # ~30 FPS grid updates

    threading.Thread(target=grid_updater, daemon=True).start()


