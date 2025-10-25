import airsim
import cv2
import numpy as np
import time
import csv
import folium
import os
import winsound   # 🔔 for audio alert (Windows only)
from collections import deque
from ai.yolov5_detector import YOLOv5PersonDetector

# CSV and map files
CSV_FILE = "survivor_detections.csv"
MAP_FILE = "survivor_map.html"

# Look-ahead depth buffer
DEPTH_HISTORY = deque(maxlen=3)

# --- Map & Logging Functions ---
def create_map():
    """Create/update survivor map from CSV"""
    if not os.path.exists(CSV_FILE):
        return

    rows = []
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            rows.append(row)

    if not rows:
        return

    center_lat, center_lon = float(rows[0][1]), float(rows[0][2])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

    for row in rows:
        t, lat, lon, alt = row
        folium.Marker(
            location=[float(lat), float(lon)],
            popup=f"Survivor @ {t} (Alt: {alt})",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    m.save(MAP_FILE)
    print(f"🗺️ Map updated → {MAP_FILE}")


def log_survivor_detection(position):
    """Log survivor detection and update map + sound alert"""
    lat, lon, alt = position.latitude, position.longitude, position.altitude
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, lat, lon, alt])

    # Print alert to console
    print(f"🚨 Survivor Detected @ {timestamp}")
    print(f"   Latitude: {lat:.6f}, Longitude: {lon:.6f}, Altitude: {alt:.2f} m")

    # Audio alert 🔊
    winsound.Beep(1000, 500)  # (frequency=1000Hz, duration=500ms)

    # Update map
    create_map()


# --- Obstacle Avoidance (Horizontal only, No Climb) ---
def obstacle_avoidance(client, depth_frame):
    h, w = depth_frame.shape
    left = np.mean(depth_frame[:, :w//3])
    center = np.mean(depth_frame[:, w//3:2*w//3])
    right = np.mean(depth_frame[:, 2*w//3:])

    DEPTH_HISTORY.append((left, center, right))

    avg_left = np.mean([d[0] for d in DEPTH_HISTORY])
    avg_center = np.mean([d[1] for d in DEPTH_HISTORY])
    avg_right = np.mean([d[2] for d in DEPTH_HISTORY])

    if avg_center < 5.0:  # Obstacle ahead
        if avg_left > avg_right:
            print("⚠️ Obstacle → Strafe Left")
            client.moveByVelocityBodyFrameAsync(2.0, -2.0, 0, 1).join()
        else:
            print("⚠️ Obstacle → Strafe Right")
            client.moveByVelocityBodyFrameAsync(2.0, 2.0, 0, 1).join()
    elif avg_center < 10.0:
        print("⚠️ Path partly blocked → Slowing Down")
        client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 1).join()
    else:
        client.moveByVelocityBodyFrameAsync(3.5, 0, 0, 1).join()  # Fast forward


# --- Main Drone Logic ---
def main():
    # Setup CSV if not exists
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("🚁 Taking off...")
    client.takeoffAsync().join()

    # Initialize YOLOv5 detector
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # Get scene + depth
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
            ])
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            # RGB image
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Depth image
            depth1d = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_frame = depth1d.reshape(responses[1].height, responses[1].width)

            # --- Fast Obstacle Avoidance (No Climb) ---
            obstacle_avoidance(client, depth_frame)

            # --- YOLO Detection ---
            annotated, detections = detector.detect(frame_bgr)
            if len(detections) > 0:
                pose = client.getGpsData().gnss.geo_point
                log_survivor_detection(pose)

            # Show detection frame
            cv2.imshow("AirSim YOLOv5 + Obstacle Avoidance", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("🛑 Program exiting, drone will hover (not forced landing).")
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
