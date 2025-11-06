import os
import time
import json
import requests
from datetime import datetime
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import pywhatkit as kit
import threading

# ------------------------------
# Config
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/shop_robbery2.mp4"
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_intrusion_shop_robb3.mp4"
zones_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/zones_config_vid_shop_rob.json"
alert_folder = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/alerts"
os.makedirs(alert_folder, exist_ok=True)

min_confidence = 0.5
trespass_class_id = 0  # YOLO person

## Telegram Bot Credentials
BOT_TOKEN = "8214541766:AAHFrh4efpd7VdTBPYQY5Mv0QYDYQ24_jY4"
CHAT_ID = "6813192996"

## WhatsApp Alert Config
WHATSAPP_NUMBER = "+8801843684994"  # replace with your number in international format


# ------------------------------
# ALERT FUNCTIONS
# ------------------------------
def send_telegram_alert(message, image_path=None):
    """Send alert message + optional image to Telegram"""
    try:
        # Send text message
        url_msg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url_msg, data={"chat_id": CHAT_ID, "text": message})

        # Send image if exists
        if image_path and os.path.exists(image_path):
            url_photo = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                payload = {"chat_id": CHAT_ID, "caption": "📸 Intrusion Snapshot"}
                requests.post(url_photo, data=payload, files={"photo": photo})
            print(f"✅ Telegram alert sent for {image_path}")

    except Exception as e:
        print(f"❌ Telegram alert failed: {e}")


def send_whatsapp_alert(message):
    """Send WhatsApp message asynchronously using pywhatkit"""
    try:
        def send_msg():
            kit.sendwhatmsg_instantly(
                phone_no=WHATSAPP_NUMBER,
                message=message,
                wait_time=10,
                tab_close=True
            )
            print("✅ WhatsApp alert sent successfully")

        # Run in a separate thread so main video loop doesn’t freeze
        threading.Thread(target=send_msg, daemon=True).start()

    except Exception as e:
        print(f"❌ WhatsApp alert failed: {e}")


# ------------------------------
# Load Multiple Zones
# ------------------------------
if not os.path.exists(zones_path):
    raise FileNotFoundError(f"Zone file not found: {zones_path}")

with open(zones_path) as f:
    zone_data = json.load(f)

zones = []
for zone in zone_data:
    zones.append({
        "name": zone["name"],
        "polygon": Polygon(zone["points"])
    })

print(f"Loaded {len(zones)} zones:")
for z in zones:
    print(" -", z["name"])

# Load YOLO model
model = YOLO("yolov8n.pt")


def process_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"🎥 Video Info: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    alerts_sent = {zone["name"]: False for zone in zones}
    intrusion_detected = False

    # Initialize counter per zone
    zone_counts = {zone["name"]: 0 for zone in zones}

    # Auto-adjust display window
    cv2.namedWindow("Multi-Zone Intrusion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Multi-Zone Intrusion Detection", 1280, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_num += 1
        print(f"Processing frame {frame_num}/{frame_count}", end="\r")

        # Reset counts per frame
        for z in zone_counts:
            zone_counts[z] = 0

        # Draw all zones
        for zone in zones:
            pts = np.array(zone["polygon"].exterior.coords, np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            label_pos = (pts[0][0] + 10, pts[0][1] - 10)
            cv2.putText(frame, zone["name"], label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Run YOLO
        results = model(frame, verbose=False, classes=[trespass_class_id])[0]

        for box in results.boxes:
            conf = float(box.conf)
            if conf < min_confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feet_x = (x1 + x2) // 2
            feet_y = y2
            point = Point(feet_x, feet_y)

            zone_name = None
            for zone in zones:
                if zone["polygon"].contains(point):
                    zone_name = zone["name"]
                    break

            if zone_name:
                zone_counts[zone_name] += 1
                intrusion_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Intruder {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 0, 255), -1)
                cv2.putText(frame, f"{zone_name}", (feet_x - 40, feet_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Alert once per zone
                if not alerts_sent[zone_name]:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(alert_folder, f"{zone_name}_intrusion_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, frame)

                    message = f"🚨 Intrusion detected in *{zone_name}*\nTime: {timestamp}\nVideo: {os.path.basename(video_path)}"

                    # Send both Telegram + WhatsApp alert
                    send_telegram_alert(message, snapshot_path)
                    send_whatsapp_alert(message)

                    alerts_sent[zone_name] = True
            else:
                # Safe zone person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 255, 0), -1)

        # ✅ Display person count per zone
        for zone in zones:
            pts = np.array(zone["polygon"].exterior.coords, np.int32)
            x, y = pts[0]
            count_text = f"Count: {zone_counts[zone['name']]}"
            cv2.putText(frame, count_text, (x + 10, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Add global warning text
        if intrusion_detected:
            cv2.putText(frame, "⚠ TRESPASS DETECTED ⚠", (width // 6, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)
        cv2.imshow("Multi-Zone Intrusion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n✅ Processing complete!")
    if intrusion_detected:
        print("🚨 Intrusions detected in video.")
    else:
        print("✅ No trespassing detected.")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    process_video()
