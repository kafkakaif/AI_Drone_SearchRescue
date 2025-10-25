# main_airsim_demo_fixed.py

from ai.yolov5_detector import YOLOv5PersonDetector
import airsim
import cv2
import numpy as np
import time
import csv
import folium
import os

CSV_FILE = "survivor_detections.csv"
MAP_FILE = "survivor_map.html"

def log_survivor_detection(position):
    """Log survivor detections to CSV"""
    lat, lon, alt = position.latitude, position.longitude, position.altitude
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, alt])
    create_map()  # update map immediately

def create_map():
    """Create a Folium map from CSV detections"""
    if not os.path.exists(CSV_FILE):
        return

    rows = []
    with open(CSV_FILE, mode="r") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            rows.append(row)

    if not rows:
        return

    # Use first detection as map center
    center_lat, center_lon = float(rows[0][1]), float(rows[0][2])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

    for row in rows:
        t, lat, lon, alt = row
        folium.Marker(
            location=[float(lat), float(lon)],
            popup=f"Survivor Detected @ {t} (Alt: {alt} m)",
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)

    m.save(MAP_FILE)
    print(f"🗺️ Map updated → {MAP_FILE}")

def main():
    # Setup CSV with headers if not exists
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("🚁 Taking off...")
    client.takeoffAsync().join()

    # Load YOLOv5 detector
    det = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # Capture scene image from drone
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run YOLOv5 detection
            annotated, detections = det.detect(frame_bgr)

            # Drone control based on detection
            if len(detections) > 0:
                pose = client.getGpsData().gnss.geo_point
                print("🚨 SURVIVOR DETECTED at:", pose)
                log_survivor_detection(pose)  # log + update map
                client.hoverAsync().join()
            else:
                # Move forward slowly
                client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 1).join()

            # Display annotated frame
            cv2.imshow("AirSim YOLOv5", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("🛬 Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
