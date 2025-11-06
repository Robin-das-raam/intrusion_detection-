import cv2
import numpy as np
import json
import os

# video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion6.mp4"
video_path = "rtsp://192.168.0.4:8080/h264_pcm.sdp"

def select_multiple_polygons(video_path, save_path="real_time_ip_zones1.json"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return []

    # Pause at frame 30
    #for real ip comment this line
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return []

    zones = []  # list of dicts: {"name": str, "points": [(x,y), ...]}
    polygon_points = []
    current_name = f"Zone_{len(zones)+1}"

    def update_display():
        temp_frame = frame.copy()
        # Draw existing zones
        for zone in zones:
            pts = np.array(zone["points"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], True, (0, 255, 0), 2)
            cx = int(np.mean([p[0] for p in zone["points"]]))
            cy = int(np.mean([p[1] for p in zone["points"]]))
            cv2.putText(temp_frame, zone["name"], (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw current polygon points
        for i, (x, y) in enumerate(polygon_points):
            cv2.circle(temp_frame, (x, y), 7, (0, 0, 255), -1)
            cv2.putText(temp_frame, str(i+1), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if len(polygon_points) > 1:
            pts = np.array(polygon_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], False, (0, 0, 255), 2)

        # Show instructions
        text_lines = [
            "Instructions:",
            "Left click: add point",
            "ENTER: save polygon",
            "'n': new zone | 'd': delete last point | 'r': reset all | ESC: exit",
            f"Current zone: {current_name}"
        ]
        y0 = 30
        for t in text_lines:
            cv2.putText(temp_frame, t, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y0 += 25

        cv2.imshow("Draw Multiple Regions", temp_frame)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            update_display()

    cv2.namedWindow("Draw Multiple Regions", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Multiple Regions", mouse_callback)
    update_display()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in [13, 10]:  # ENTER
            if len(polygon_points) >= 3:
                zones.append({"name": current_name, "points": polygon_points.copy()})
                print(f"✅ Saved {current_name} with {len(polygon_points)} points.")
                polygon_points.clear()
                current_name = f"Zone_{len(zones)+1}"
                update_display()
            else:
                print("⚠️ Need at least 3 points for a zone.")
        elif key == ord('n'):  # skip to next zone
            if len(polygon_points) >= 3:
                zones.append({"name": current_name, "points": polygon_points.copy()})
                polygon_points.clear()
            current_name = f"Zone_{len(zones)+1}"
            update_display()
        elif key == ord('d'):  # delete last point
            if polygon_points:
                polygon_points.pop()
            update_display()
        elif key == ord('r'):  # reset all
            zones.clear()
            polygon_points.clear()
            current_name = "Zone_1"
            update_display()
        elif key == 27:  # ESC → exit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save zones to JSON
    if zones:
        with open(save_path, "w") as f:
            json.dump(zones, f, indent=4)
        print(f"\n💾 Saved {len(zones)} zones to {os.path.abspath(save_path)}")
    else:
        print("No zones saved.")

    return zones


# Example usage
zones = select_multiple_polygons(video_path)
print("Final zones:", zones)
