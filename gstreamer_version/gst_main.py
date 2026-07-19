import time
import threading
import numpy as np
import cv2
import math
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, Response

from configs import CAMERA_URLS, RESIZED_DIM, STOP_EVENT
from gst_infer_app import inference_loop, latest_frames, frame_lock
from gst_streamer import GstH264Streamer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI()

N_CAM = len(CAMERA_URLS)
COLS = 2


def round_up_to_stride(x, stride=32):
    return int(math.ceil(x / stride) * stride)


def even(x: int) -> int:
    x = int(x)
    return max(2, x - (x % 2))


def build_grid(frames, cols=COLS):
    # Filter out None, but keep positions if we want a fixed layout.
    # For now, just build from whatever we have.
    valid = [f for f in frames if f is not None]
    if not valid:
        return None
    h, w, _ = valid[0].shape
    rows = (N_CAM + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for idx in range(N_CAM):
        if idx < len(frames) and frames[idx] is not None:
            r = idx // cols
            c = idx % cols
            grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = frames[idx]
    return grid


# ------------------------------------------------------------------
# Stream geometry
# ------------------------------------------------------------------
small_w = round_up_to_stride(RESIZED_DIM[0], 32)
small_h = round_up_to_stride(RESIZED_DIM[1], 32)
rows = (N_CAM + COLS - 1) // COLS
grid_w = COLS * small_w
grid_h = rows * small_h

SCALE_OUT = 0.8
STREAM_W = even(int(grid_w * SCALE_OUT))
STREAM_H = even(int(grid_h * SCALE_OUT))
FPS_OUT = 30

logger.info("Grid canvas: %dx%d | Stream output: %dx%d", grid_w, grid_h, STREAM_W, STREAM_H)

gst_streamer = GstH264Streamer(STREAM_W, STREAM_H, fps=FPS_OUT, bitrate_kbps=4000)


# ------------------------------------------------------------------
# Startup threads
# ------------------------------------------------------------------
@app.on_event("startup")
def start_threads():
    threading.Thread(target=inference_loop, daemon=True).start()
    logger.info("Inference loop started.")

    def grid_to_gstreamer():
        push_count = 0
        while not STOP_EVENT.is_set():
            with frame_lock:
                frames = [latest_frames.get(i) for i in range(N_CAM)]

            # DEBUG: log every 5 seconds whether we have frames
            if int(time.time()) % 5 == 0:
                available = [i for i, f in enumerate(frames) if f is not None]
                logger.info("Available cameras: %s | latest_frames keys: %s", available, list(latest_frames.keys()))

            grid = build_grid(frames, cols=COLS)
            if grid is None:
                time.sleep(0.05)
                continue

            if grid.shape[1] != STREAM_W or grid.shape[0] != STREAM_H:
                grid = cv2.resize(grid, (STREAM_W, STREAM_H), interpolation=cv2.INTER_AREA)

            try:
                gst_streamer.push_frame(grid)
                push_count += 1
                if push_count % 30 == 0:
                    logger.info("Pushed %d frames to encoder", push_count)
            except Exception as exc:
                logger.error("push_frame failed: %s", exc)

            time.sleep(1.0 / FPS_OUT)

    threading.Thread(target=grid_to_gstreamer, daemon=True).start()
    logger.info("Grid feeder started.")


# ------------------------------------------------------------------
# HTTP routes
# ------------------------------------------------------------------
@app.get("/")
def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mpegts.js@latest"></script>
</head>
<body style="margin:0;background:black;display:flex;justify-content:center;align-items:center;height:100vh;">
  <video id="v" controls autoplay muted playsinline style="width:90vw;max-width:1200px;"></video>
  <script>
    var player = mpegts.createPlayer({
      type: 'mpegts',
      url: '/live_h264',
      isLive: true,
      liveBufferLatencyChasing: true
    });
    player.attachMediaElement(document.getElementById('v'));
    player.load();
    player.play();
  </script>
</body>
</html>
""")


@app.get("/live_h264")
def live_h264():
    return StreamingResponse(
        gst_streamer.get_chunks(),
        media_type="video/mp2t",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.head("/live_h264")
def live_h264_head():
    return Response(status_code=200, media_type="video/mp2t")


@app.on_event("shutdown")
def on_shutdown():
    logger.info("Shutdown signal received.")
    STOP_EVENT.set()
    gst_streamer.stop()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)