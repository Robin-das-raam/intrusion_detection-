import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

input_img_path = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/images"
output = "/home/robinpc/Desktop/FastApi_prac/intrusion_detection/embeddings/robin_face.npy"
# Initialize ArcFace model
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1)  # -1 = CPU, use 0 if GPU available

def register_face(folder_path, output_file):
    embeddings = []
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping {img_name}, cannot load.")
            continue

        faces = app.get(img)
        if len(faces) == 0:
            print(f"⚠️ No face detected in {img_name}")
            continue

        # Take the first detected face
        emb = faces[0].normed_embedding
        embeddings.append(emb)

    if len(embeddings) == 0:
        print("❌ No faces registered. Exiting.")
        return

    # Average embedding across all images
    mean_embedding = np.mean(embeddings, axis=0)

    # Save embedding
    np.save(output_file, mean_embedding)
    print(f"✅ Registered {len(embeddings)} faces. Saved to {output_file}")

# Example usage
register_face(input_img_path, output)
