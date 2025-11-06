import os
import time
import subprocess
from datetime import datetime
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
import pywhatkit as kit   # For WhatsApp alerts

from whatsappalert import send_whatsapp_alert_nonblocking

# ------------------------------
# Config
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing_vid1.mp4"          
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_intrusion1.mp4"   
polygon_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/roi_polygon2.npy"    
alert_folder = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/alerts/"
os.makedirs(alert_folder, exist_ok=True)

min_confidence = 0.5
trespass_class_id = 0  # YOLO class for person


# ------------------------------
# WhatsApp Alert Function
# ------------------------------
def ensure_copyq_running():
    """Ensure the CopyQ clipboard manager is running silently."""
    try:
        result = subprocess.run(["pgrep", "-x", "copyq"], stdout=subprocess.PIPE)
        if result.returncode != 0:
            subprocess.Popen(
                ["copyq"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            time.sleep(2)  # give it time to start
    except Exception as e:
        print("⚠️ Could not verify or start copyq:", e)

# def send_whatsapp_alert(receiver, image_path, message):
#     """Send image + message via WhatsApp using pywhatkit."""
#     ensure_copyq_running()
#     try:
#         print("📤 Sending WhatsApp alert...")
#         kit.sendwhats_image(
#             receiver=receiver,
#             img_path=image_path,
#             caption=message,
#             wait_time=15,     # adjust if slow internet
#             tab_close=True
#         )
#         print("✅ WhatsApp alert sent successfully.")
#     except Exception as e:
#         print(f"❌ Failed to send WhatsApp alert: {e}")


# ------------------------------
# Intrusion Detection Logic
# ------------------------------
polygon_points = np.load(polygon_path, allow_pickle=True)
trespass_polygon = Polygon(polygon_points)

model = YOLO("yolov8n.pt")   # you can replace with yolov11 or your custom model


def process_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    alert_sent = False  
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    trespass_detected = False
    
    # WhatsApp receiver number (must include country code)
    WHATSAPP_NUMBER = "+8801843684994"   # <-- replace with your number

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_num += 1
        print(f"Processing frame {frame_num}/{frame_count}", end='\r')
        
        # Draw restricted zone
        cv2.polylines(frame, [np.array(polygon_points, dtype=np.int32)], True, (0, 0, 255), 2)
        cv2.putText(frame, "Restricted Area", 
                    (polygon_points[0][0], polygon_points[0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # YOLO inference
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
                
                # Draw red bounding box for intruder
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Intrusion {conf:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 0, 255), -1)

                if not alert_sent:
                    snapshot_frame = frame.copy()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    snapshot_path = os.path.join(alert_folder, f"intrusion_{timestamp}.jpg")
                    cv2.imwrite(snapshot_path, snapshot_frame)
                    print(f"📸 Snapshot saved: {snapshot_path}")

                    alert_msg = f"🚨 Intrusion Detected!\nTime: {timestamp}"
                    send_whatsapp_alert_nonblocking(WHATSAPP_NUMBER,alert_msg,snapshot_path)
                    alert_sent = True
            else:
                # Green bounding box for normal person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 255, 0), -1)
        
        if current_trespass:
            cv2.putText(frame, "TRESPASSING DETECTED!", (width//6, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        out.write(frame)
        cv2.imshow("Intrusion Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if trespass_detected:
        print("\n🚨 Trespassing detected in video!")
    else:
        print("\n✅ No trespassing detected.")
    print(f"🎥 Output saved to: {output_path}")


if __name__ == "__main__":
    process_video()
