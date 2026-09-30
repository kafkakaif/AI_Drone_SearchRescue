import airsim
import cv2
import numpy as np
import time
import csv
import folium
import os
import winsound
import threading
from collections import deque
from ai.yolov5_detector import YOLOv5PersonDetector
CSV_FILE = "survivor_detections.csv"
MAP_FILE = "survivor_map.html"
DEPTH_HISTORY = deque(maxlen=2)
latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
detection_lock = threading.Lock()
running = True
last_detection_time = 0
DETECTION_COOLDOWN = 5
def create_map():
    """Create/update survivor map from CSV"""
    if not os.path.exists(CSV_FILE):
        return
    rows = []
    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                rows.append(row)
    if not rows:
        return
    center_lat = float(rows[0][1])
    center_lon = float(rows[0][2])
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18
    )
    for row in rows:
        t, lat, lon, alt = row
        folium.Marker(
            location=[
                float(lat),
                float(lon)
            ],
            popup=f"Survivor @ {t} (Alt: {alt} m)",
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(m)
    m.save(MAP_FILE)
    print(f"🗺️ Map updated → {MAP_FILE}")
def log_survivor_detection(position):
    """Log person detection and update map"""
    global last_detection_time
    current_time = time.time()
    with detection_lock:
        if current_time - last_detection_time < DETECTION_COOLDOWN:
            return False
        last_detection_time = current_time
    lat = position.latitude
    lon = position.longitude
    alt = position.altitude
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            lat,
            lon,
            alt
        ])
    print("\n🚨 PERSON / SURVIVOR DETECTED!")
    print(f"⏰ Time: {timestamp}")
    print(f"📍 Latitude: {lat:.6f}")
    print(f"📍 Longitude: {lon:.6f}")
    print(f"ارتفاع Altitude: {alt:.2f} m")
    create_map()
    return True
def yolo_worker(detector):
    global latest_frame
    global latest_detections
    global running
    while running:
        if latest_frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            frame_copy = latest_frame.copy()
        resized = cv2.resize(
            frame_copy,
            (416, 416)
        )
        annotated, detections = detector.detect(resized)
        with detection_lock:
            latest_detections = detections
        if len(detections) > 0:
            try:
                winsound.Beep(1000, 150)
            except RuntimeError:
                pass
def obstacle_avoidance(client, depth_frame):
    h, w = depth_frame.shape
    band = depth_frame[
        h // 3: 2 * h // 3,
        :
    ]
    left = float(
        np.min(
            band[:, :w // 3]
        )
    )
    center = float(
        np.min(
            band[:, w // 3:2 * w // 3]
        )
    )
    right = float(
        np.min(
            band[:, 2 * w // 3:]
        )
    )
    DEPTH_HISTORY.append(center)
    avg_center = float(
        np.mean(DEPTH_HISTORY)
    )
    base_speed = 7.0
    safety = 4.5
    if avg_center < safety:
        client.moveByVelocityBodyFrameAsync(
            0.0,
            0.0,
            0.0,
            0.05
        )
        if left > right:
            client.rotateByYawRateAsync(
                -150.0,
                0.25
            )
        else:
            client.rotateByYawRateAsync(
                150.0,
                0.25
            )
        return
    client.moveByVelocityBodyFrameAsync(
        base_speed,
        0.0,
        0.0,
        0.08
    )
def main():
    global latest_frame
    global latest_detections
    global running
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Time",
                "Latitude",
                "Longitude",
                "Altitude (m)"
            ])
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("🚁 Taking off...")
    client.takeoffAsync().join()
    detector = YOLOv5PersonDetector(
        "yolov5s",
        0.45
    )
    yolo_thread = threading.Thread(
        target=yolo_worker,
        args=(detector,),
        daemon=True
    )
    yolo_thread.start()
    print("👤 Person detection started")
    print("🗺️ Survivor map logging started")
    print("🚀 Drone autonomous mode started")
    try:
        while True:
            responses = client.simGetImages([
                airsim.ImageRequest(
                    "0",
                    airsim.ImageType.Scene,
                    False,
                    False
                ),
                airsim.ImageRequest(
                    "0",
                    airsim.ImageType.DepthPerspective,
                    True
                )
            ])
            if not responses:
                continue
            # Check image validity
            if responses[0].height == 0:
                continue
            img1d = np.frombuffer(
                responses[0].image_data_uint8,
                dtype=np.uint8
            )
            frame = img1d.reshape(
                responses[0].height,
                responses[0].width,
                3
            )
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR
            )
            with frame_lock:
                latest_frame = frame_bgr.copy()
            depth1d = np.array(
                responses[1].image_data_float,
                dtype=np.float32
            )
            depth_frame = depth1d.reshape(
                responses[1].height,
                responses[1].width
            )
            obstacle_avoidance(
                client,
                depth_frame
            )
            with detection_lock:
                detections_copy = list(latest_detections)
            # If person detected
            if len(detections_copy) > 0:
                # Get drone GPS position
                gps_data = client.getGpsData()
                position = gps_data.gnss.geo_point
                # Log detection + update map
                was_logged = log_survivor_detection(
                    position
                )
                if was_logged:
                    try:
                        winsound.Beep(
                            1000,
                            500
                        )
                    except RuntimeError:
                        pass
            cv2.imshow(
                "🚀 Ultra-Fast AI Drone",
                frame_bgr
            )
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print("🛑 Stopping drone program...")
        running = False
        yolo_thread.join(timeout=2)
        client.moveByVelocityBodyFrameAsync(
            0,
            0,
            0,
            0.5
        )
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()
        print("✅ Program stopped safely")
if __name__ == "__main__":
    main()
