import airsim
import time
import cv2
import numpy as np
import os
import csv
import folium

# -------------------- CONFIG --------------------
CSV_FILE = "survivor_detections.csv"
MAP_FILE = "survivor_map.html"
DRONE_NAME = ""  # leave blank to use default vehicle

# YOLOv5 import (assuming ai/yolov5_detector.py exists)
from ai.yolov5_detector import YOLOv5PersonDetector

# -------------------- SETUP --------------------
print("Connecting to AirSim...")
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, DRONE_NAME)
client.armDisarm(True)
print("Connected!")

detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

# -------------------- HELPER FUNCTIONS --------------------
def create_map():
    """Create/update survivor map from CSV"""
    if not os.path.exists(CSV_FILE):
        return

    rows = []
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            rows.append(row)

    if not rows:
        return

    center_lat, center_lon = float(rows[0][1]), float(rows[0][2])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

    for row in rows:
        t, x, y, z = row
        folium.Marker(
            location=[float(x), float(y)],
            popup=f"Survivor @ {t} (Alt: {z})",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)

    m.save(MAP_FILE)
    print(f"🗺️ Map updated → {MAP_FILE}")

def log_survivor_detection(position):
    """Log survivor detection and update map"""
    lat, lon, alt = position.x_val, position.y_val, position.z_val
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, alt])
    create_map()

def get_front_depth():
    """Return front depth in meters"""
    responses = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
    ])
    depth_img = np.array(responses[0].image_data_float, dtype=np.float32)
    if depth_img.size == 0:
        return 1000  # if no data, assume far
    return np.mean(depth_img)

def get_rgb_frame():
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    return img1d.reshape(responses[0].height, responses[0].width, 3)

# -------------------- MISSION --------------------
try:
    print("🚁 Taking off...")
    client.takeoffAsync().join()
    print("✅ Drone in air. Starting mission...")

    for step in range(50):  # adjust number of steps
        depth = get_front_depth()
        frame = get_rgb_frame()
        annotated, detections = detector.detect(frame)

        # show camera
        cv2.imshow("Camera", annotated)
        cv2.waitKey(1)

        if len(detections) > 0:
            print("⚠️ Survivor detected! Hovering...")
            client.hoverAsync().join()
            log_survivor_detection(client.getMultirotorState().kinematics_estimated.position)
        elif depth < 5.0:
            print(f"🚧 Obstacle detected at {depth:.2f}m, stopping...")
            client.hoverAsync().join()
        else:
            client.moveByVelocityAsync(5, 0, 0, 1).join()  # move forward at 5 m/s

    print("✅ Drone mission ended.")

finally:
    print("Landing drone...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    cv2.destroyAllWindows()
