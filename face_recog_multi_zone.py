import os
import json
import cv2
import numpy as np
import requests
from datetime import datetime
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ------------------------------
# Configurations
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion6.mp4"
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_intrusion6.mp4"
zones_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/zones_config_intru6.json"
alert_folder = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/alerts"
os.makedirs(alert_folder, exist_ok=True)

min_confidence = 0.5
trespass_class_id = 0  # YOLO class ID for "person"

# Telegram Bot (optional)
# BOT_TOKEN = "your_bot_token"
# CHAT_ID = "your_chat_id"

# ------------------------------
# Face Recognition Setup
# ------------------------------
from insightface.app import FaceAnalysis

face_app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
face_app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode

# Registered face embeddings
registered_faces = {
    "Robin": np.load("/home/robinpc/Desktop/FastApi_prac/intrusion_detection/embeddings/robin_face.npy"),
    # Add more embeddings if needed
}

FACE_MATCH_THRESHOLD = 0.45  # Lower = stricter (0.4–0.6 typical)


def is_known_face(face_embedding):
    """Compare face embedding with registered faces"""
    for name, known_emb in registered_faces.items():
        sim = np.dot(face_embedding, known_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_emb))
        if sim > (1 - FACE_MATCH_THRESHOLD):  # cosine similarity check
            return name
    return None


# ------------------------------
# Optional: Telegram Alert
# ------------------------------
def send_telegram_alert(message, image_path=None):
    try:
        url_msg = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url_msg, data={"chat_id": CHAT_ID, "text": message})
        if image_path and os.path.exists(image_path):
            url_photo = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
            with open(image_path, "rb") as photo:
                payload = {"chat_id": CHAT_ID, "caption": "📸 Intrusion Snapshot"}
                requests.post(url_photo, data=payload, files={"photo": photo})
    except Exception as e:
        print(f"❌ Telegram alert failed: {e}")


# ------------------------------
# Load Zones
# ------------------------------
if not os.path.exists(zones_path):
    raise FileNotFoundError(f"Zone file not found: {zones_path}")

with open(zones_path) as f:
    zone_data = json.load(f)

zones = [{"name": z["name"], "polygon": Polygon(z["points"])} for z in zone_data]
print(f"✅ Loaded {len(zones)} zones: {[z['name'] for z in zones]}")

# Load YOLO model
model = YOLO("yolov8n.pt")

# ------------------------------
# Main Video Processing
# ------------------------------
def process_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error opening video file")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Video Info: {width}x{height} @ {fps:.2f} FPS, {frame_count} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    alerts_sent = {z["name"]: False for z in zones}
    intrusion_detected = False
    zone_counts = {z["name"]: 0 for z in zones}

    cv2.namedWindow("Face-Aware Intrusion Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face-Aware Intrusion Detection", 1280, 720)

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_num += 1
        print(f"Processing frame {frame_num}/{frame_count}", end="\r")

        # Reset zone counts
        for z in zone_counts:
            zone_counts[z] = 0

        # Draw zone outlines
        for zone in zones:
            pts = np.array(zone["polygon"].exterior.coords, np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)
            cv2.putText(frame, zone["name"], (pts[0][0] + 10, pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Run YOLO person detection
        results = model(frame, verbose=False, classes=[trespass_class_id])[0]

        for box in results.boxes:
            conf = float(box.conf)
            if conf < min_confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_crop = frame[y1:y2, x1:x2]

            # Run face recognition
            faces = face_app.get(person_crop)
            recognized_name = None
            for face in faces:
                emb = face['embedding']
                match_name = is_known_face(emb)
                if match_name:
                    recognized_name = match_name
                    break

            feet_x, feet_y = (x1 + x2) // 2, y2
            point = Point(feet_x, feet_y)
            zone_name = next((z["name"] for z in zones if z["polygon"].contains(point)), None)

            # Draw detections
            color = (0, 255, 0) if recognized_name else (0, 0, 255)
            label = recognized_name if recognized_name else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (feet_x, feet_y), 6, color, -1)

            # Handle intrusion logic
            if zone_name:
                zone_counts[zone_name] += 1

                if not recognized_name:  # unknown person only
                    intrusion_detected = True
                    if not alerts_sent[zone_name]:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        snapshot_path = os.path.join(alert_folder, f"{zone_name}_intrusion_{timestamp}.jpg")
                        cv2.imwrite(snapshot_path, frame)
                        message = f"🚨 Intrusion detected in *{zone_name}*\nTime: {timestamp}"
                        send_telegram_alert(message, snapshot_path)
                        alerts_sent[zone_name] = True

        # Display person counts
        for z in zones:
            pts = np.array(z["polygon"].exterior.coords, np.int32)
            x, y = pts[0]
            text = f"Count: {zone_counts[z['name']]}"
            cv2.putText(frame, text, (x + 10, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if intrusion_detected:
            cv2.putText(frame, "⚠ TRESPASS DETECTED ⚠", (width // 6, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        out.write(frame)
        cv2.imshow("Face-Aware Intrusion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\n✅ Processing complete!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    process_video()
