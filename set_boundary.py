
import cv2
import numpy as np

video_path = "/home/robinpc/Desktop/FastApi_prac/shop_lifting/shoplifting_dataset/Shop DataSet/shop lifters/shop_lifter_14.mp4"


def select_polygon(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return []

    # Pause at frame 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        return []

    polygon_points = []

    def update_display():
        temp_frame = frame.copy()
        # Draw points
        for i, (x, y) in enumerate(polygon_points):
            cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(temp_frame, str(i+1), (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        # Draw polygon
        if len(polygon_points) > 1:
            pts = np.array(polygon_points, np.int32).reshape((-1,1,2))
            cv2.polylines(temp_frame, [pts], False, (0,255,0), 2)
        cv2.imshow("Draw Polygon - ENTER to finish", temp_frame)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            polygon_points.append((x, y))
            update_display()

    cv2.namedWindow("Draw Polygon - ENTER to finish", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Polygon - ENTER to finish", mouse_callback)
    update_display()

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == 13 or key == 10:  # ENTER
            if len(polygon_points) >= 3:
                np.save("roi_polygon.npy", np.array(polygon_points))
                print("Polygon saved:", polygon_points)
                break
            else:
                print("Need at least 3 points")
        elif key == ord('d'):
            if polygon_points:
                polygon_points.pop()
                update_display()
        elif key == ord('c'):
            polygon_points = []
            update_display()
        elif key == 27:  # ESC
            polygon_points = []
            break

    cap.release()
    cv2.destroyAllWindows()
    return polygon_points

# Example usage
points = select_polygon(video_path)
print("Selected polygon points:", points)
