import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Config
# ------------------------------
video_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/testing videos/multiple_intrusion6.mp4"  # input video path
registered_embedding_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/embeddings/robin_face.npy"
# output_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/output_face_recognition.mp4"
threshold = 0.35  # similarity threshold

# Load registered embedding
registered_embedding = np.load(registered_embedding_path)

# Initialize ArcFace
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)  # CPU mode

# Open video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Error opening video file")

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}/{frame_count}", end="\r")

    faces = app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Face embedding
        emb = face.normed_embedding.reshape(1, -1)
        reg_emb = registered_embedding.reshape(1, -1)

        sim = cosine_similarity(emb, reg_emb)[0][0]

        if sim >= threshold:
            label = f"KNOWN ({sim:.2f})"
            color = (0, 255, 0)
        else:
            label = f"UNKNOWN ({sim:.2f})"
            color = (0, 0, 255)

        # Draw results
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show live video
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
# out.release()

print(f"\n✅ Processing complete! Output saved at {output_path}")
