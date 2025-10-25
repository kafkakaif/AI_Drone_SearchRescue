"""
Quick demo: Run person detection on a video (no simulator required).
Put a file at data/sample.mp4 or use webcam if file is missing.
"""
import cv2
import os
from ai.detector import PersonDetector

VIDEO_PATH = os.path.join("data", "sample.mp4")

def open_source():
    if os.path.exists(VIDEO_PATH):
        return cv2.VideoCapture(VIDEO_PATH), "video"
    # fallback to webcam
    return cv2.VideoCapture(0), "webcam"

def main():
    det = PersonDetector(conf=0.5)
    cap, src_type = open_source()
    if not cap.isOpened():
        print("ERROR: Could not open video or webcam.")
        return

    print(f"[INFO] Running detection on {src_type}... Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated, detections = det.detect(frame)
        cv2.imshow("YOLOv8 Person Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
