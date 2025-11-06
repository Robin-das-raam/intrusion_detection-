import cv2
import numpy as np
import os

# ======== CONFIG ========
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion8.mp4"
save_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/zones_and_anchors8.npy"
# ========================

def select_zones_and_anchors(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error opening video file.")
        return {}

    # Jump to the middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    if not ret:
        print("❌ Error reading middle frame.")
        return {}

    zones = {}
    temp_frame = frame.copy()
    cv2.namedWindow("Zone Setup", cv2.WINDOW_NORMAL)

    zone_idx = 1
    polygon_points = []
    anchor_points = []
    mode = "polygon"  # or "anchor"

    def update_display():
        disp = frame.copy()

        # draw existing zones
        for name, data in zones.items():
            pts = np.array(data["polygon"], np.int32).reshape((-1, 1, 2))
            cv2.polylines(disp, [pts], True, (0, 255, 255), 2)
            for (ax, ay) in data["anchors"]:
                cv2.circle(disp, (ax, ay), 6, (255, 0, 0), -1)
            cv2.putText(disp, name, tuple(data["polygon"][0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # draw current zone
        if polygon_points:
            cv2.polylines(disp, [np.array(polygon_points, np.int32)], False, (0, 255, 0), 2)
            for (x, y) in polygon_points:
                cv2.circle(disp, (x, y), 6, (0, 255, 0), -1)
        for (x, y) in anchor_points:
            cv2.circle(disp, (x, y), 6, (255, 0, 0), -1)

        cv2.putText(disp, f"Mode: {mode.upper()} (Press 'm' to switch)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Zone Setup", disp)

    def mouse_callback(event, x, y, flags, param):
        nonlocal polygon_points, anchor_points, mode
        if event == cv2.EVENT_LBUTTONDOWN:
            if mode == "polygon":
                polygon_points.append((x, y))
            elif mode == "anchor":
                anchor_points.append((x, y))
            update_display()

    cv2.setMouseCallback("Zone Setup", mouse_callback)
    update_display()

    print("\n🟢 Controls:")
    print(" - Left Click: add point (polygon or anchor depending on mode)")
    print(" - Press 'm': toggle between polygon/anchor mode")
    print(" - Press 'n': save current zone & start new one")
    print(" - Press 'd': delete last point")
    print(" - Press 'q': finish all zones and save\n")

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('m'):  # toggle mode
            mode = "anchor" if mode == "polygon" else "polygon"
            update_display()

        elif key == ord('d'):  # delete last point
            if mode == "polygon" and polygon_points:
                polygon_points.pop()
            elif mode == "anchor" and anchor_points:
                anchor_points.pop()
            update_display()

        elif key == ord('n'):  # next zone
            if len(polygon_points) >= 3:
                zone_name = f"zone_{zone_idx}"
                zones[zone_name] = {
                    "polygon": polygon_points.copy(),
                    "anchors": anchor_points.copy()
                }
                print(f"✅ Saved {zone_name}")
                zone_idx += 1
                polygon_points.clear()
                anchor_points.clear()
                mode = "polygon"
                update_display()
            else:
                print("⚠️ Need at least 3 points for polygon")

        elif key == ord('q'):  # quit and save all
            if len(polygon_points) >= 3:
                zone_name = f"zone_{zone_idx}"
                zones[zone_name] = {
                    "polygon": polygon_points.copy(),
                    "anchors": anchor_points.copy()
                }
            break

    cap.release()
    cv2.destroyAllWindows()
    np.save(save_path, zones)
    print(f"\n💾 All zones saved to: {save_path}")
    return zones


if __name__ == "__main__":
    result = select_zones_and_anchors(video_path)

    print("\n✅ Zones saved. Checking details:")
    for zone_name, data in result.items():
        polygon = data["polygon"]
        anchors = data["anchors"]
        print(f"{zone_name}:")
        print(f"  Polygon points ({len(polygon)} points): {polygon}")
        print(f"  Anchors ({len(anchors)} points): {anchors}")
        print(f"  Polygon type: {type(polygon)}, Anchors type: {type(anchors)}\n")

    print("Zones saved:", result.keys())