import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO

# ------------------------------
# Config
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/shop_lifting/shoplifting_dataset/Shop DataSet/shop lifters/shop_lifter_14.mp4"          # input video
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_intrusion.mp4"   # output video
polygon_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/roi_polygon.npy"    # polygon boundary saved earlier
min_confidence = 0.5
trespass_class_id = 0  # YOLO class for person

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
            else:
                # Safe person (green)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0, 255, 0), -1) 
        
        if current_trespass:
            cv2.putText(frame, "TRESPASSING DETECTED!", (width//6, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    if trespass_detected:
        print("\nTrespassing detected in video!")
    else:
        print("\nNo trespassing detected")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    process_video()
