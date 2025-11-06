import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO  # or your own detection model

# ---- Load zones safely ----
zone_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/zones_and_anchors8.npy"
loaded_zones = np.load(zone_path, allow_pickle=True)

# Unwrap 0-d array if necessary
if isinstance(loaded_zones, np.ndarray) and loaded_zones.ndim == 0:
    loaded_zones = loaded_zones.item()  # now it’s the dict

# Extract polygons and anchors
polygons = [Polygon(data["polygon"]) for data in loaded_zones.values()]
anchors = [data["anchors"] for data in loaded_zones.values()]
zone_names = list(loaded_zones.keys())

# ---- Define restricted zones by index (example) ----
restricted_zones = [0, 1]  # mark first two zones as restricted

# ---- Load video ----
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion7.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("❌ Error opening video file")

# ---- Load YOLO model ----
model = YOLO("yolov8n.pt")  # replace with your own model if needed

# ---- Main loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model(frame, verbose=False)
    detections = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    # Reset zone counts
    zone_counts = [0] * len(polygons)

    for i, box in enumerate(detections):
        cls_id = int(classes[i])
        if cls_id != 0:  # only person class
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2)//2, y2  # use feet point
        point = Point(cx, cy)

        # Check which zone the person is in
        for idx, poly in enumerate(polygons):
            if poly.contains(point):
                zone_counts[idx] += 1

                # Draw bounding box
                color = (0, 255, 0)  # safe
                if idx in restricted_zones:
                    color = (0, 0, 255)
                    cv2.putText(frame, "⚠ Intrusion", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                break

        # Draw center point
        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

    # Draw zones and counts
    for idx, poly_data in enumerate(loaded_zones.values()):
        poly_pts = np.array(poly_data["polygon"], np.int32).reshape((-1, 1, 2))
        color = (255, 255, 0)  # normal zone
        if idx in restricted_zones:
            color = (0, 0, 255)
        cv2.polylines(frame, [poly_pts], isClosed=True, color=color, thickness=2)
        cv2.putText(frame,
                    f"{zone_names[idx]}: {zone_counts[idx]} person(s)",
                    (poly_pts[0][0][0] + 10, poly_pts[0][0][1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Display
    cv2.imshow("Multi-Zone Intrusion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
