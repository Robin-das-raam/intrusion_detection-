
# ---- Configuration ----
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion4.mp4"
output_folder = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/my_img"  

import cv2
import os
from ultralytics import YOLO

# ---------------------- CONFIG ----------------------
model_path = "yolov8n.pt"  # Use 'yolov8n.pt' or 'yolov8s.pt' depending on speed/accuracy
min_conf = 0.5  # Minimum confidence to accept detection
frame_skip = 8  # Process every nth frame to reduce redundancy
person_class_id = 0  # COCO class ID for "person"

# ----------------------------------------------------
os.makedirs(output_folder, exist_ok=True)
model = YOLO(model_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Error: Cannot open video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"🎥 FPS: {fps}, Total Frames: {total_frames}")

frame_num = 0
saved_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_num % frame_skip != 0:
        frame_num += 1
        continue

    results = model(frame, verbose=False)[0]  # Run YOLO inference
    person_detected = False

    for box in results.boxes:
        cls = int(box.cls)
        conf = float(box.conf)

        if cls == person_class_id and conf >= min_conf:
            person_detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Crop and save the person
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue  # skip invalid crops

            # Optional: filter too small detections
            h, w = cropped.shape[:2]
            if h < 100 or w < 100:
                continue

            filename = os.path.join(output_folder, f"person_{frame_num:05d}.jpg")
            cv2.imwrite(filename, cropped)
            saved_count += 1

            # Visualize detections
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display progress and preview
    if person_detected:
        cv2.imshow("Person Detection", frame)
    else:
        cv2.imshow("Person Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
print(f"✅ Done! Saved {saved_count} cropped person images in '{output_folder}'")
