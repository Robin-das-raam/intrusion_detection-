import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from numpy.linalg import norm

# ------------------------------
# Config
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion5.mp4"
polygon_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/zones_config_intru5.json"
output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_intrusion_face5.mp4"

min_confidence = 0.5
trespass_class_id = 0  # YOLO class for person

# Registered faces embeddings
registered_faces = {
    "Robin": np.load("/home/robinpc/Desktop/FastApi_prac/intrusion_detection/embeddings/robin_face.npy"),
    # Add more registered faces if needed
}

# ------------------------------
# Load Models
# ------------------------------
# YOLO model
model = YOLO("yolov8n.pt")

# Face recognition model
face_app = FaceAnalysis(allowed_modules=['detection','recognition'])
face_app.prepare(ctx_id=-1, det_size=(640,640))  # CPU mode

# ------------------------------
# Load ROI polygon
# ------------------------------
polygon_points = np.load(polygon_path, allow_pickle=True)
trespass_polygon = Polygon(polygon_points)

# ------------------------------
# Helper Functions
# ------------------------------
def is_registered(face_embedding, registered_faces, threshold=0.5):
    """
    Check if face embedding matches any registered face
    """
    for name, emb in registered_faces.items():
        sim = np.dot(face_embedding, emb) / (norm(face_embedding) * norm(emb))
        if sim > threshold:
            return True, name
    return False, None

# ------------------------------
# Main Video Processing
# ------------------------------
def process_video():
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer to save output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trespass_detected = False

    # Create full screen window
    cv2.namedWindow("Intrusion + Face Recognition", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Intrusion + Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Draw ROI polygon
        cv2.polylines(frame, [np.array(polygon_points, dtype=np.int32)], True, (0,0,255), 2)
        cv2.putText(frame, "Restricted Area", (polygon_points[0][0], polygon_points[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # YOLO inference for person detection
        results = model(frame, verbose=False, classes=[trespass_class_id])[0]
        
        current_trespass = False
        
        for box in results.boxes:
            conf = float(box.conf)
            if conf < min_confidence:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            feet_x, feet_y = (x1 + x2) // 2, y2
            point = Point(feet_x, feet_y)

            # Crop detected person for face recognition
            person_crop = frame[y1:y2, x1:x2]
            faces = face_app.get(person_crop)

            recognized = False
            face_name = None
            if len(faces) > 0:
                # Take first detected face in person crop
                face_embedding = faces[0].embedding
                recognized, face_name = is_registered(face_embedding, registered_faces)

            if trespass_polygon.contains(point) and not recognized:
                current_trespass = True
                trespass_detected = True
                # Draw intruder (red)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f'Intrusion {conf:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0,0,255), -1)
            else:
                # Recognized or safe person (green)
                label = face_name if recognized else f'Person {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.circle(frame, (feet_x, feet_y), 8, (0,255,0), -1)

        if current_trespass:
            cv2.putText(frame, "TRESPASSING DETECTED!", (width//6, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        
        # Show live frame
        cv2.imshow("Intrusion + Face Recognition", frame)
        # Save frame
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if trespass_detected:
        print("\nTrespassing detected in video!")
    else:
        print("\nNo trespassing detected")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    process_video()
