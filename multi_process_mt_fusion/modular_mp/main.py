import time
import threading
import numpy as np
import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from inference import inference_loop

app = FastAPI()

# placeholders (will be filled at startup)
# latest_frames = {}
# frame_lock = None

def build_grid(frames, cols=2):
    """
    frames: list of same-sized frames (H, W)
    """
    if not frames:
        return None

    h, w, _ = frames[0].shape
    rows = (len(frames) + cols - 1) // cols

    grid = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)

    for idx, frame in enumerate(frames):
        r = idx // cols
        c = idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = frame

    return grid

@app.get("/live_grid")
def live_grid():
    def generator():
        while True:
            with frame_lock:
                frames = list(latest_frames.values())

            if not frames:
                time.sleep(0.05)
                continue

            grid = build_grid(frames, cols=2)

            ret, buffer = cv2.imencode(".jpg", grid)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes() +
                b"\r\n"
            )

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )



# @app.on_event("startup")
# def startup_event():
#     threading.Thread(target=inference_loop, daemon=True).start()

@app.on_event("startup")
def start():
    global inference_loop, latest_frames, frame_lock

    from inference import inference_loop as _loop
    from inference import latest_frames as _frames
    from inference import frame_lock as _lock

    inference_loop = _loop
    latest_frames = _frames
    frame_lock = _lock

    threading.Thread(target=inference_loop, daemon=True).start()
