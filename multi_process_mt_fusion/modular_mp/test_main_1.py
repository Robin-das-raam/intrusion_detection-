import time
import threading
import subprocess
import numpy as np
import cv2
import math

from fastapi import FastAPI
from fastapi.responses import Response

from fastapi.responses import StreamingResponse, HTMLResponse

from configs import CAMERA_URLS, RESIZED_DIM

app = FastAPI()

# Optional MJPEG
try:
    from turbojpeg import TurboJPEG
    jpeg = TurboJPEG()
except Exception:
    jpeg = None

# ---------------- Shared grid buffers ----------------
grid_buffer = [None, None]
current_grid_idx = 0
grid_lock = threading.Lock()
grid_seq = 0

N_CAM = len(CAMERA_URLS)
COLS = 2

# Inference shared state
from update_inference_1 import inference_loop, latest_frames, frame_lock


# ---------------- Utils ----------------
def round_up_to_stride(x, stride=32):
    return int(math.ceil(x / stride) * stride)

def build_grid(frames, cols=COLS):
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

def even(x: int) -> int:
    x = int(x)
    return max(2, x - (x % 2))


# ---------------- H.264 (ffmpeg) streaming ----------------
def start_ffmpeg_h264(stream_w: int, stream_h: int, fps_out: int):
    """
    stdin: rawvideo BGR24 frames
    stdout: mpegts with H.264
    """
    # Start with x264 for reliability
    cmd = [
    "ffmpeg", "-loglevel", "error",
    "-fflags", "nobuffer",
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-s", f"{stream_w}x{stream_h}",
    "-r", str(fps_out),
    "-i", "pipe:0",
    "-an",

    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-profile:v", "baseline",
    "-level", "3.0",
    "-bf", "0",

    "-g", "15",
    "-keyint_min", "15",
    "-sc_threshold", "0",

    "-pix_fmt", "yuv420p",

    "-flush_packets", "1",
    "-mpegts_flags", "+resend_headers+pat_pmt_at_frames",

    "-f", "mpegts",
    "-muxdelay", "0",
    "-muxpreload", "0",
    "pipe:1",
]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # keep stderr for debug
        bufsize=0
    )


@app.on_event("startup")
def start_threads():
    # Start inference thread
    threading.Thread(target=inference_loop, daemon=True).start()

    # Build/refresh grid periodically
    def grid_updater():
        global grid_buffer, current_grid_idx, grid_seq
        while True:
            with frame_lock:
                frames = [latest_frames.get(i) for i in range(N_CAM)]

            if any(f is None for f in frames):
                time.sleep(0.01)
                continue

            new_grid = build_grid(frames, cols=COLS)

            with grid_lock:
                current_grid_idx = 1 - current_grid_idx
                grid_buffer[current_grid_idx] = new_grid
                grid_seq += 1

            time.sleep(0.03)  # ~33 FPS grid rebuild cadence (tune if needed)

    threading.Thread(target=grid_updater, daemon=True).start()


@app.get("/")
def index():
    return HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/mpegts.js@latest"></script>
</head>
<body style="margin:0;background:black;display:flex;flex-direction:column;align-items:center;justify-content:center;">
  <video id="v" controls autoplay muted playsinline style="width:100vw;max-width:1200px;"></video>
  <script>
    var player = mpegts.createPlayer({
      type: 'mpegts',
      url: 'http://127.0.0.1:7000/live_h264',
      isLive: true
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
    """
    H.264 MPEG-TS stream over HTTP.
    """
    # Compute deterministic sizes from RESIZED_DIM + grid layout
    # RESIZED_DIM is (W,H) for your cv2.resize usage
    small_w = round_up_to_stride(RESIZED_DIM[0], 32)
    small_h = round_up_to_stride(RESIZED_DIM[1], 32)

    rows = (N_CAM + COLS - 1) // COLS
    grid_w = COLS * small_w
    grid_h = rows * small_h

    SCALE_OUT = 0.8
    FPS_OUT = 30

    stream_w = even(grid_w * SCALE_OUT)
    stream_h = even(grid_h * SCALE_OUT)

    p = start_ffmpeg_h264(stream_w, stream_h, FPS_OUT)

    stop = threading.Event()

    def feeder():
        local_last_seq = -1
        last_frame_resized = None

        next_t = time.perf_counter()
        while not stop.is_set():
            # rate-limit feeding
            now = time.perf_counter()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += 1.0 / FPS_OUT

            with grid_lock:
                grid = grid_buffer[current_grid_idx]
                seq = grid_seq

            if grid is None:
                continue

            if seq != local_last_seq or last_frame_resized is None:
                last_frame_resized = cv2.resize(
                    grid, (stream_w, stream_h),
                    interpolation=cv2.INTER_AREA
                )
                local_last_seq = seq

            try:
                p.stdin.write(last_frame_resized.tobytes())
            except (BrokenPipeError, OSError):
                stop.set()
                break

    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    def gen():
        try:
            while True:
                chunk = p.stdout.read(4096)
                if not chunk:
                    break
                yield chunk
        finally:
            stop.set()
            try:
                if p.stdin:
                    p.stdin.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="video/mp2t")

@app.head("/live_h264")
def live_h264_head():
    return Response(status_code=200, media_type="video/mp2t")