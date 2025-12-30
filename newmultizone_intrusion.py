import os
import time 
import json
import cv2
import numpy as np
from datetime import datetime
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

video_path = "rtsp://admin:Doer2022%24%23@202.125.77.226:554/Streaming/Channels/501"
# video_path = "http://192.168.0.4:8080/video"


zones_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/zones/real_time_ip_zones2.json"
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output/real_time_test_v10_1.mp4"

min_confidence = 0.5
class_id = 0

# Load Restriced zones

if not os.path.exists(zones_path):
    raise FileNotFoundError(f"Zone file not found: {zones_path}")

with open(zones_path) as f:
    zone_data = json.load(f)

zones = [
    {"name":z["name"],"polygon":Polygon(z["points"])}
    for z in zone_data
]

print(f"Loaded {len(zones)} Restricted Zones")

# Load YOLOV10 Model

model = YOLO("yolov10n.pt")

def process_video():
    # 🚀 Enable low-latency streaming with FFmpeg options
    gst_str = (
    "souphttpsrc location=http://192.168.0.4:8080/video ! "
    "jpegdec ! videoconvert ! "
    "appsink sync=false drop=true max-buffers=1"
    )
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    # cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Failed to open stream")

    # Video resolution from stream
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    print(f"Stream: {width}x{height} @ {fps:.2f} FPS")

    # Output video Writer
    fourcc = cv2. VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width,height))

    cv2.namedWindow("YOLOv10 Intrusion System", cv2.WINDOW_NORMAL)

    prev_time = time.time()

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        # ret, frame = cap.read()
        if not ret:
            print("No Frame.. Retrying...")
            

        # FPS measurement
        now = time.time()
        fps_val = 1 / (now - prev_time)
        prev_time = now

        # Reset counts
        zone_counts = {z["name"]: 0 for z in zones}

        # Draw zones
        for zone in zones:
            pts = np.array(zone["polygon"].exterior.coords,np.int32)
            cv2.polylines(frame,[pts], True, (255,0,0),2)

        infer_frame = cv2.resize(frame, (640, 480))
        results = model(infer_frame, verbose=False, classes = [class_id])[0]

        intrusion_detected = False

        for box in results.boxes:
            conf = float(box.conf)
            if conf < min_confidence:
                continue
            x1,y1,x2,y2 =  map(int, box.xyxy[0])
            feet =  Point((x1 + x2) // 2, y2)
            # check if inside restricted zone
            zone_name = None
            for zone in zones:
                if zone["polygon"].contains(feet):
                    zone_name = zone["name"]
                    zone_counts[zone_name] +=1
                    intrusion_detected = True
                    break

            # Draw detection
            color = (0, 0, 255) if zone_name else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Display counts
        for zone in zones:
            pts = np.array(zone["polygon"].exterior.coords, np.int32)
            x, y = pts[0]
            cv2.putText(frame,
                        f"{zone_counts[zone['name']]}",
                        (x + 10, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2)
            
        # Add warning text
        if intrusion_detected:
            cv2.putText(frame, "⚠ INTRUSION DETECTED ⚠",
                        (40, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3)

        cv2.putText(frame, f"FPS: {fps_val:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2)

        out.write(frame)
        cv2.imshow("YOLOv10 Intrusion System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Streaming stopped!")


if __name__ == "__main__":
    process_video()

        

