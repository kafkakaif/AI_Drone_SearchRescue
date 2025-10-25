from ai.yolov5_detector import YOLOv5PersonDetector
import airsim
import cv2
import numpy as np
import time
import csv
import os

CSV_FILE = "survivor_detections.csv"

def log_survivor(position):
    """Log detected survivor GPS coordinates"""
    lat, lon, alt = position.latitude, position.longitude, position.altitude
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, alt])
    print(f"📍 Logged survivor at: lat={lat}, lon={lon}, alt={alt}")

def main():
    # Create CSV if not exists
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    print("🚀 Drone took off!")

    # Load YOLOv5 detector
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # Capture front camera image
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run YOLO detection
            annotated, detections = detector.detect(frame_bgr)

            if len(detections) > 0:
                pose = client.getGpsData().gnss.geo_point
                print("🚨 SURVIVOR DETECTED!")
                log_survivor(pose)
                client.hoverAsync().join()
            else:
                # Move forward slowly
                client.moveByVelocityBodyFrameAsync(1.0, 0, 0, 1).join()

            # Show camera feed
            cv2.imshow("Blocks AirSim YOLOv5", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("🛬 Landing drone...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
