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

    from update_inference import inference_loop, latest_frames, frame_lock

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


#### testing 

# import time
# import threading
# import numpy as np
# import cv2
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse, JSONResponse
# from turbojpeg import TurboJPEG
# from collections import deque

# app = FastAPI()
# jpeg = TurboJPEG()

# # ----------------------------------------------------------------------
# # Double‑buffer + metrics
# # ----------------------------------------------------------------------
# grid_buffer = [None, None]
# current_grid_idx = 0
# grid_lock = threading.Lock()

# # Metrics (EMA – very cheap, no list growth)
# metrics_lock = threading.Lock()
# metrics = {
#     "grid_build_ms": 0.0,   # time spent in build_grid (in updater thread)
#     "encode_ms": 0.0,       # time spent in jpeg.encode (in stream thread)
#     "stream_loop_ms": 0.0,  # total time per iteration of the generator
#     "alpha": 0.2,           # smoothing factor (0.2 ≈ last 5 samples)
# }

# def update_ema(name, new_val):
#     """Thread‑safe EMA update."""
#     with metrics_lock:
#         alpha = metrics["alpha"]
#         old = metrics[name]
#         metrics[name] = alpha * new_val + (1 - alpha) * old

# def get_metrics_snapshot():
#     with metrics_lock:
#         return {k: round(v, 2) for k, v in metrics.items() if not k == "alpha"}

# # ----------------------------------------------------------------------
# # Core logic
# # ----------------------------------------------------------------------
# def build_grid(frames, cols=2):
#     if not frames:
#         return None
#     h, w, _ = frames[0].shape
#     rows = (len(frames) + cols - 1) // cols
#     grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
#     for idx, frame in enumerate(frames):
#         r = idx // cols
#         c = idx % cols
#         grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frame
#     return grid

# def update_grid(frames):
#     """Runs in background thread – NEVER blocks the HTTP client."""
#     global grid_buffer, current_grid_idx

#     t0 = time.perf_counter()
#     new_grid = build_grid(frames, cols=2)
#     build_ms = (time.perf_counter() - t0) * 1000
#     update_ema("grid_build_ms", build_ms)

#     if new_grid is None:
#         return

#     with grid_lock:
#         current_grid_idx = 1 - current_grid_idx
#         grid_buffer[current_grid_idx] = new_grid

# @app.get("/live_grid")
# def live_grid():
#     def generator():
#         last_grid_idx = -1
#         while True:
#             loop_t0 = time.perf_counter()

#             with grid_lock:
#                 if grid_buffer[current_grid_idx] is None:
#                     time.sleep(0.005)
#                     continue
#                 if current_grid_idx == last_grid_idx:
#                     # No new frame -> tiny sleep to avoid busy‑spin, but **latency is zero**
#                     time.sleep(0.001)
#                     continue
#                 last_grid_idx = current_grid_idx
#                 grid = grid_buffer[current_grid_idx]

#             # ----- Only thing that happens in the critical request path: FAST encode -----
#             t_enc0 = time.perf_counter()
#             jpeg_bytes = jpeg.encode(grid, quality=80)   # 3‑5× faster than cv2
#             enc_ms = (time.perf_counter() - t_enc0) * 1000
#             update_ema("encode_ms", enc_ms)

#             yield (
#                 b"--frame\r\n"
#                 b"Content-Type: image/jpeg\r\n\r\n"
#                 + jpeg_bytes +
#                 b"\r\n"
#             )

#             # Track total per‑iteration latency seen by the streaming loop
#             total_ms = (time.perf_counter() - loop_t0) * 1000
#             update_ema("stream_loop_ms", total_ms)

#     return StreamingResponse(
#         generator(),
#         media_type="multipart/x-mixed-replace; boundary=frame"
#     )

# @app.get("/stats")
# def stats():
#     """Returns live latency numbers in **milliseconds**."""
#     snap = get_metrics_snapshot()
#     # Add a helpful derived metric
#     snap["total_latency_reduction_vs_naive"] = round(
#         (snap["stream_loop_ms"] / (snap["grid_build_ms"] + snap["encode_ms"] + 1e-6)) * 100, 1
#     ) if snap["grid_build_ms"] > 0 else 0
#     return JSONResponse(snap)

# # ----------------------------------------------------------------------
# # Startup – pulls inference loop & starts grid updater
# # ----------------------------------------------------------------------
# @app.on_event("startup")
# def start():
#     global latest_frames, frame_lock
#     from inference import inference_loop, latest_frames, frame_lock

#     # 1. Start model inference (produces frames)
#     threading.Thread(target=inference_loop, daemon=True).start()

#     # 2. Start grid builder (decoupled from streaming!)
#     def grid_updater():
#         while True:
#             with frame_lock:
#                 frames = list(latest_frames.values())
#             update_grid(frames)
#             time.sleep(0.02)  # ~50 Hz grid refresh – independent of client pull rate
#     threading.Thread(target=grid_updater, daemon=True).start()