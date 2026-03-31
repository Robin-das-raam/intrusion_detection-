import torch 
import multiprocessing as mp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STOP_EVENT = mp.Event()

#### MODEL #############
MODEL_PATH = "yolov8n.pt"
CONF_THRESH = 0.4
RESIZED_DIM = (640,420)
QUEUE_SIZE = 4

#### CAMERAS #############

CAMERA_URLS = [
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501",
    "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/101",
]

ZONES_PATHS = [
    "/home/doer/Desktop/workstation/intrusion_detection-/office_ip_cam2_zones.json",
    "/home/doer/Desktop/workstation/intrusion_detection-/office_ip_cam1_zones.json",
]

# ===============================
# ALERTS
# ===============================
BOT_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
ALERT_COOLDOWN = 30
