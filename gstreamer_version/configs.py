import torch 
import math
import os
from dotenv import load_dotenv
import multiprocessing as mp

STOP_EVENT = mp.Event()

# 🔑 CRITICAL FIX: Find .env relative to THIS file's location
# This ensures it works even if you run scripts from different folders
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Debug: Print if .env loaded (Optional, helps troubleshooting)
# print(f"[Config] .env loaded from: {dotenv_path}")
# print(f"[Config] CAMERA_URL_1 found: {'YES' if os.getenv('CAMERA_URL_1') else 'NO'}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The original resolution of the video frame you used to draw your zones.
# For example, if you took a 1920x1080 snapshot to draw your zones, use that here.
ZONES_SRC_DIM = (1920, 1080)

#### MODEL #############
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
RESIZED_DIM = (640,480)
# RESIZED_WIDTH = 640
# RESIZED_HEIGHT = 480

# Original desired dimensions
_RESIZED_W, _RESIZED_H = (640, 480) # Using a more standard size for example

# # Function to ensure dimensions are compatible with model stride (usually 32)
def _round_up_to_stride(x, stride=32):
    return int(math.ceil(x / stride) * stride)

# Stride-aligned dimensions that will be used in the pipeline
RESIZED_WIDTH = _round_up_to_stride(_RESIZED_W)
RESIZED_HEIGHT = _round_up_to_stride(_RESIZED_H)

# 
QUEUE_SIZE = 4

#### CAMERAS #############
CAMERA_URLS = [
    os.getenv("CAMERA_URL_1", "default_url_1"),
    os.getenv("CAMERA_URL_2", "default_url_2")
]
# Check if URLs were loaded correctly
if CAMERA_URLS[0] == "default_url_1" or CAMERA_URLS[1] == "default_url_2":
    print("WARNING: Could not find CAMERA_URL_1 or CAMERA_URL_2 in .env file.")
    print("Please create a .env file and add the URLs.")

ZONES_PATHS = [
    "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam2_zones.json",
    "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/office_ip_cam1_zones.json",
]

# ===============================
# ALERTS
# ===============================
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID = os.getenv("CHAT_ID", "")
ALERT_COOLDOWN = 30

STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8001