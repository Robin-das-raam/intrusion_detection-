# import cv2
# import time
# import threading
# import sys
# from configs import CAMERA_URLS

# class TestCameraThread:
#     def __init__(self, src):
#         # Refined pipeline for OpenCV GStreamer backend
#         pipeline = (
#             f"rtspsrc location={src} latency=100 ! "
#             f"rtph264depay ! "
#             f"h264parse ! "
#             f"avdec_h264 ! "
#             f"videoconvert ! "
#             f"video/x-raw, format=BGR ! "
#             f"appsink"
#         )
        
#         print(f"Opening pipeline...")
#         self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
#         if not self.cap.isOpened():
#             print("ERROR: Failed to open GStreamer pipeline!")
#             self.running = False
#             return

#         # Give stream time to initialize
#         time.sleep(1)
#         self.ret, self.frame = self.cap.read()
#         self.running = True
#         threading.Thread(target=self.update, daemon=True).start()

#     def update(self):
#         while self.running:
#             ret, frame = self.cap.read()
#             if ret:
#                 self.ret, self.frame = ret, frame
#             else:
#                 time.sleep(0.1)

#     def read(self):
#         if self.ret and self.frame is not None:
#             return self.frame
#         return None

#     def stop(self):
#         self.running = False
#         self.cap.release()

# if __name__ == "__main__":
#     if not CAMERA_URLS:
#         print("No camera URLs found in configs.py")
#         sys.exit(1)

#     print(f"Testing Camera 0")
#     cam = TestCameraThread(CAMERA_URLS[0])

#     if not cam.running:
#         print("Failed to initialize camera.")
#         sys.exit(1)

#     print("Stream opened! Press 'q' to quit")
#     fps_counter = 0
#     last_time = time.time()

#     while True:
#         frame = cam.read()
#         if frame is not None:
#             fps_counter += 1
#             if time.time() - last_time >= 1.0:
#                 print(f"FPS: {fps_counter}")
#                 fps_counter = 0
#                 last_time = time.time()

#             cv2.imshow("GStreamer Test", frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         else:
#             print("Waiting for frames...")
#             time.sleep(0.5)

#     cam.stop()
#     cv2.destroyAllWindows()
#     print("Test finished.")

import cv2
from configs import CAMERA_URLS  # load_dotenv() is already called inside configs

print("=" * 50)
print("OpenCV GStreamer Support Check")
print("=" * 50)

# 1. Check if CAMERA_URLS has data
if not CAMERA_URLS or not CAMERA_URLS[0]:
    print("\n❌ ERROR: CAMERA_URLS is empty!")
    print("   Check .env file exists in the same folder as configs.py")
    exit(1)

url = CAMERA_URLS[0]
print(f"\n[Debug] First URL: {url[:50]}...")

# 2. Build pipeline
pipeline = (
    f'rtspsrc location="{url}" latency=100 ! '
    f'rtph264depay ! '
    f'avdec_h264 ! '
    f'videoconvert ! '
    f'appsink'
)

print(f"\nPipeline String:\n{pipeline}\n")

# 3. Open capture
print("Opening VideoCapture with GStreamer backend...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# 4. Check if opened BEFORE calling getBackendName()
if cap.isOpened():
    print(f"✅ Pipeline Opened Successfully")
    print(f"Backend Used: {cap.getBackendName()}")
    
    print("Attempting to read frame...")
    ret, frame = cap.read()
    if ret:
        print(f"✅ SUCCESS! Frame shape: {frame.shape}")
    else:
        print("❌ FAILED: Could not read frame (check network/password)")
    
    cap.release()
else:
    print("❌ FAILED: Could not open pipeline")

print("=" * 50)