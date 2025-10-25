from ai.yolov5_detector import YOLOv5PersonDetector
import airsim
import cv2
import numpy as np
import time
import csv
import os
import folium

# Files for logging
CSV_FILE = "survivor_detections.csv"
MAP_FILE = "survivor_map.html"

def log_survivor_detection(position):
    """Log survivor detections to CSV and update map."""
    lat, lon, alt = position.latitude, position.longitude, position.altitude
    with open(CSV_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), lat, lon, alt])
    create_map()

def create_map():
    """Generate map from logged detections."""
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

    # Center map on first detection
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
    # Setup CSV header if first time
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

    # Initialize YOLOv5 detector
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # Get front camera image
            responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            # Convert to numpy array
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Detect persons
            annotated, detections = detector.detect(frame_bgr)

            if len(detections) > 0:
                pose = client.getGpsData().gnss.geo_point
                print("🚨 SURVIVOR DETECTED at:", pose)
                log_survivor_detection(pose)
                client.hoverAsync().join()
            else:
                client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 1).join()

            # Show detection
            cv2.imshow("AirSim YOLOv5", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
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
    """Create survivor map from CSV detections"""
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
    # Setup CSV with headers
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

    # connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    # detector
    det = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            annotated, detections = det.detect(frame_bgr)

            if len(detections) > 0:
                pose = client.getGpsData().gnss.geo_point
                print("🚨 SURVIVOR DETECTED at:", pose)
                log_survivor_detection(pose)  # log + update map
                client.hoverAsync().join()
            else:
                client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 1).join()

            cv2.imshow("AirSim YOLOv5", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
from ai.yolov5_detector import YOLOv5PersonDetector
import airsim
import cv2
import numpy as np
import time
import csv


def main():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    # Load YOLOv5 detector
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    # Open CSV file to log detections
    log_file = open("survivor_detections.csv", mode="w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["Time", "X", "Y", "Z"])  # CSV header

    try:
        while True:
            # Get image from AirSim
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            # Convert AirSim image to numpy
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)

            # AirSim usually returns RGB → convert to BGR (for OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run YOLO detection
            annotated, detections = detector.detect(frame_bgr)

            # Drone behavior
            if len(detections) > 0:
                print("👀 Survivor detected! Hovering...")
                client.hoverAsync().join()

                # Get drone GPS position
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                x, y, z = pos.x_val, po_
from ai.yolov5_detector import YOLOv5PersonDetector
import airsim
import cv2
import numpy as np
import time


def main():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()

    # Load YOLOv5 detector
    detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # Get image from AirSim
            responses = client.simGetImages(
                [airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)]
            )
            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            # Convert AirSim image to numpy
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)

            # AirSim usually returns RGB → convert to BGR (for OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run YOLO detection
            annotated, detections = detector.detect(frame_bgr)

            # Drone behavior
            if len(detections) > 0:
                print("👀 Survivor detected! Hovering...")
                client.hoverAsync().join()
            else:
                print("No survivor, moving forward...")
                client.moveByVelocityBodyFrameAsync(1.0, 0, 0, 1).join()

            # Show the annotated frame
            cv2.imshow("AirSim YOLOv5", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
"""
Integrated AirSim demo:
 - Connect to simulator
 - Takeoff
 - Loop: get RGB + Depth
   * Controller decides forward/turn using depth
   * YOLO detects persons from RGB
   * If person detected: hover & log pose
 - Press 'q' to exit, then land.
"""
import cv2
import time
from ai.yolov5_detector import YOLOv5PersonDetector
from ai.detector import PersonDetector
from drone.airsim_agent import AirSimAgent
from drone.controller import AvoidanceController

def main():
    # Init modules
    det = PersonDetector(conf=0.5)
    ctrl = AvoidanceController(depth_threshold_m=5.0, window_ratio=0.2)
    agent = AirSimAgent()

    print("[INFO] Taking off...")
    agent.takeoff(altitude=-3.0)
    time.sleep(1.0)

    print("[INFO] Entering control loop. Press 'q' to land and exit.")
    while True:
        rgb, depth = agent.get_rgb_depth()
        # Decide motion from depth
        action = ctrl.decide(depth)

        # Run detection
        annotated, dets = det.detect(rgb)

        # If any person detected => hover & print pose
        if len(dets) > 0:
            agent.hover(duration=1.0)
            x, y, z = agent.get_pose()
            cv2.putText(
                annotated, "SURVIVOR DETECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA
            )
            cv2.putText(
                annotated, f"Pose (x={x:.1f}, y={y:.1f}, z={z:.1f})",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
            )
        else:
            # Simple policy
            if action == "forward":
                agent.move_forward(vel=2.0, duration=0.5)
            else:
                agent.turn(yaw_rate_deg=30.0, duration=0.7)

        cv2.imshow("AI Drone: Detect + Avoid (AirSim)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Landing...")
    agent.land()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# main_airsim_demo.py

from ai.yolov5_detector import YOLOv5PersonDetector
import airsim
import cv2
import numpy as np
import time


def main():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("Taking off...")
    client.takeoffAsync().join()

    # Initialize YOLOv5 detector
    det = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

    try:
        while True:
            # Get image from drone's front camera
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])

            if not responses or responses[0].height == 0:
                time.sleep(0.1)
                continue

            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height, responses[0].width, 3)

            # Convert AirSim's RGB → OpenCV BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Run YOLOv5 detection
            annotated, detections = det.detect(frame_bgr)

            # Control logic
            if len(detections) > 0:
                print("SURVIVOR DETECTED; hovering. Pose:", client.getMultirotorState().kinematics_estimated.position)
                client.hoverAsync().join()
            else:
                # Move forward slowly
                client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 1).join()

            # Show detection window
            cv2.imshow("AirSim YOLOv5", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        print("Landing...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
