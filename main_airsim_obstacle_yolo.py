import airsim
import cv2
import numpy as np
import time
import csv
import os
from ai.yolov5_detector import YOLOv5PersonDetector

CSV_FILE = "survivor_detections.csv"

def log_survivor_detection(position):
    """Log survivor detections to CSV"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "X", "Y", "Z"])
    
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"),
                         position.x_val, position.y_val, position.z_val])

def main():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("🚁 Taking off...")
    client.takeoffAsync().join()

    # YOLOv5 detector
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # --- Get images ---
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
            ])
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            # RGB image
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            rgb_frame = img1d.reshape(responses[0].height, responses[0].width, 3)
            rgb_bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

            # Depth image
            depth_image = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_image = depth_image.reshape(responses[1].height, responses[1].width)

            # --- YOLOv5 detection ---
            annotated, detections = detector.detect(rgb_bgr)

            # --- Obstacle detection ---
            front_region = depth_image[200:400, 300:500]  # central area
            min_dist = np.min(front_region)

            # --- Drone decision logic ---
            if len(detections) > 0:
                print("🚨 SURVIVOR DETECTED! Hovering...")
                client.hoverAsync().join()
                pos = client.getMultirotorState().kinematics_estimated.position
                log_survivor_detection(pos)
            elif min_dist < 3.0:
                print(f"⚠ Obstacle detected ahead ({min_dist:.2f}m)! Moving sideways...")
                client.moveByVelocityBodyFrameAsync(0, 1.5, 0, 1).join()  # sideways
            else:
                client.moveByVelocityBodyFrameAsync(2, 0, 0, 1).join()  # move forward

            # Show annotated frame
            cv2.imshow("AirSim Drone YOLO+Depth", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("🛬 Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
