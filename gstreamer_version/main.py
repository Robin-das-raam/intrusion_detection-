# main.py
import time
import cv2
from up_inference_core import CameraThread, overlay_info
from configs import CAMERA_URLS

if __name__ == "__main__":
    # Test First Camera Only
    cam = CameraThread(CAMERA_URLS[0], cam_id=0)
    
    if not cam.ret:
        print("Failed to start. Check .env passwords and network.")
        exit(1)

    print("Press 'q' to quit")
    while True:
        frame = cam.read()
        if frame is not None:
            frame = overlay_info(frame, 0, 0, 0)
            cv2.imshow("Intrusion Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.01)
    
    cam.stop()
    cv2.destroyAllWindows()