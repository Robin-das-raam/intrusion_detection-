#video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion7.mp4"

import cv2
import numpy as np
from shapely.geometry import Polygon, Point
from ultralytics import YOLO

# ------------------------
# CONFIG
# ------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion2.mp4"
model = YOLO("yolov8n.pt")

# ------------------------
# STEP 1: DRAW POLYGON ON FIRST FRAME
# ------------------------
def draw_polygon(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return None
    polygon_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(polygon_points) > 1:
                cv2.line(frame, polygon_points[-2], polygon_points[-1], (0, 255, 0), 2)
            cv2.imshow("Draw Restricted Zone (Press ENTER when done)", frame)

    cv2.imshow("Draw Restricted Zone (Press ENTER when done)", frame)
    cv2.setMouseCallback("Draw Restricted Zone (Press ENTER when done)", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # ENTER
            break
    cv2.destroyAllWindows()
    cap.release()
    return np.array(polygon_points, dtype=np.int32)


# ------------------------
# STEP 2: SELECT DOOR REGION TO TRACK (ANCHOR)
# ------------------------
def select_door_region(frame):
    r = cv2.selectROI("Select Door Region", frame, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = map(int, r)
    return x, y, w, h


# ------------------------
# STEP 3: PROCESS VIDEO WITH TRACKED ROI
# ------------------------
def process_with_anchored_zone(video_path, zone_points, door_region):
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    if not ret:
        print("Error opening video")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    (x, y, w, h) = door_region
    door_patch = old_gray[y:y+h, x:x+w]

    # Extract good features to track from door patch
    p0 = cv2.goodFeaturesToTrack(door_patch, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if p0 is not None:
        p0[:, 0, 0] += x
        p0[:, 0, 1] += y

    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    zone_polygon = Polygon(zone_points)
    zone_color = (0, 0, 255)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) > 0:
                dx, dy = np.mean(good_new - good_old, axis=0)
                zone_points = np.int32(zone_points + [dx, dy])
                zone_polygon = Polygon(zone_points)

            # update for next iteration
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        # Draw ROI
        cv2.polylines(frame, [zone_points], True, zone_color, 2)
        cv2.putText(frame, "Grounded Restricted Zone", (zone_points[0][0], zone_points[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)

        # Run YOLO detection
        results = model(frame, verbose=False, classes=[0])[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feet_x, feet_y = (x1 + x2) // 2, y2
            color = (0, 255, 0)
            if zone_polygon.contains(Point(feet_x, feet_y)):
                color = (0, 0, 255)
                cv2.putText(frame, "🚨 Intrusion Detected!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (feet_x, feet_y), 4, color, -1)

        cv2.imshow("Anchored Zone Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ------------------------
# MAIN PIPELINE
# ------------------------
if __name__ == "__main__":
    # Step 1: draw zone
    zone_points = draw_polygon(video_path)
    if zone_points is None:
        exit()

    # Step 2: choose door patch to track
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    door_region = select_door_region(frame)

    # Step 3: run tracking + detection
    process_with_anchored_zone(video_path, zone_points, door_region)
