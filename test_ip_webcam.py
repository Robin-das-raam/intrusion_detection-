import cv2
url = "http://192.168.0.4:8080/video"  # replace
cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)   # try with FFMPEG backend
if not cap.isOpened():
    print("Failed to open capture")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No frame, retrying...")
        cv2.waitKey(500)
        continue
    cv2.imshow("phone_live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
