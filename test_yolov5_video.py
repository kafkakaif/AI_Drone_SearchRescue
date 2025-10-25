import cv2
from ai.yolov5_detector import YOLOv5PersonDetector

def main():
    # Load YOLOv5 model
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    # Open your test video
    cap = cv2.VideoCapture("sample.mp4")  # put your video file here

    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        # Run YOLO detection
        annotated_frame, detections = detector.detect(frame)

        # Show detections
        cv2.imshow("YOLOv5 Test", annotated_frame)

        # Print detection details
        if len(detections) > 0:
            print("Detected:", detections)

        # Exit if 'q' pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# test_yolov5_video.py
import cv2
from ai.yolov5_detector import YOLOv5PersonDetector
import os

VIDEO_PATH = os.path.join("data", "swimming.mp4")  # your file

det = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

cap = cv2.VideoCapture(VIDEO_PATH if os.path.exists(VIDEO_PATH) else 0)
if not cap.isOpened():
    print("Cannot open video/webcam")
    exit()

while True:
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    annotated, detections = det.detect(frame)
    # print detections for debug
    if detections:
        print("Detected persons:", len(detections))
    cv2.imshow("YOLOv5 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
