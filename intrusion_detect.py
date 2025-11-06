import os
import time
import requests
from datetime import datetime
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# ------------------------------
# Config
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/shop_lifter_8.mp4"          # input video
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_intrusion1.mp4"   # output video
polygon_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/roi_polygon1.npy"    # polygon boundary saved earlier
alert_folder = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/alerts"
os.makedirs(alert_folder, exist_ok=True)


min_confidence = 0.5
trespass_class_id = 0  # YOLO class for person

## Telegram Bot Credentials
BOT_TOKEN = "8214541766:AAHFrh4efpd7VdTBPYQY5Mv0QYDYQ24_jY4"
CHAT_ID = "6813192996"

def send_telegram_alert(message, image_path=None):
    """Send alert msg to telegram with image"""

    try:
        # Send text 
        url_msg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url_msg, data={"chat_id":CHAT_ID, "text":"message"})

        # Send image if available
        if image_path and os.path.exists(image_path):
            url_photo = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                payload = {"chat_id": CHAT_ID, "caption": "📸 Intrusion Snapshot"}
                files = {"photo": photo}
                resp = requests.post(url_photo, data=payload, files=files)
                if resp.status_code == 200:
                    print("✅ Telegram image sent successfully.")
                else:
                    print("❌ Failed to send image:", resp.text)
            
    except Exception as e:
        print(f"❌ Telegram alert failed: {e}")

# Load polygon points
polygon_points = np.load(polygon_path, allow_pickle=True)
trespass_polygon = Polygon(polygon_points)

# Load YOLO model
model = YOLO("yolov8n.pt")   # you can replace with yolov11 or custom later


def process_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    alert_sent = False  
    
    # Get video props
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    trespass_detected = False

    # last_alert_time = 0
    # alert_cooldown = 10   # seconds
    
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_num += 1
        print(f"Processing frame {frame_num}/{frame_count}", end='\r')
        
        # Draw polygon
        cv2.polylines(frame, [np.array(polygon_points, dtype=np.int32)], 
                      True, (0, 0, 255), 2)
        cv2.putText(frame, "Restricted Area", 
                   (polygon_points[0][0], polygon_points[0][1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Run YOLO inference
        results = model(frame, verbose=False, classes=[trespass_class_id])[0]
        
        current_trespass = False
        
        for box in results.boxes:
            conf = float(box.conf)
            if conf < min_confidence:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feet_x = (x1 + x2) // 2
            feet_y = y2
            
            point = Point(feet_x, feet_y)
            if trespass_polygon.contains(point):
                current_trespass = True
                trespass_detected = True
                
                # Draw intruder (red)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Intrusion {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 0, 255), -1)

                # Save snapshot & send alert once per intrusion
                if not alert_sent:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(alert_folder, f"intrusion_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, frame)
                    print(f"📸 Snapshot saved: {snapshot_path}")

                    message = f"🚨 Intrusion Detected!\nTime: {timestamp}\nVideo: {os.path.basename(video_path)}"
                    send_telegram_alert(message, snapshot_path)
                    alert_sent = True
            else:
                # Safe person (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 255, 0), -1) 
        
        if current_trespass:
            cv2.putText(frame, "TRESPASSING DETECTED!", (width//6, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            
        # Show live frame
        cv2.imshow("Intrusion Recognition", frame)
        # Save frame
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # out.write(frame)
    
    cap.release()
    out.release()
    
    if trespass_detected:
        print("\nTrespassing detected in video!")
    else:
        print("\nNo trespassing detected")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    process_video()
