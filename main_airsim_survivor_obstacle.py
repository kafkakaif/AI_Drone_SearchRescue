# # import airsim
# # import cv2
# # import numpy as np
# # import time
# # import csv
# # import folium
# # import os
# # import winsound   # 🔔 for audio alert (Windows only)
# # from collections import deque
# # from ai.yolov5_detector import YOLOv5PersonDetector

# # # CSV and map files
# # CSV_FILE = "survivor_detections.csv"
# # MAP_FILE = "survivor_map.html"

# # # Look-ahead depth buffer
# # DEPTH_HISTORY = deque(maxlen=3)

# # # --- Map & Logging Functions ---
# # def create_map():
# #     """Create/update survivor map from CSV"""
# #     if not os.path.exists(CSV_FILE):
# #         return

# #     rows = []
# #     with open(CSV_FILE, "r") as f:
# #         reader = csv.reader(f)
# #         next(reader, None)  # skip header
# #         for row in reader:
# #             rows.append(row)

# #     if not rows:
# #         return

# #     center_lat, center_lon = float(rows[0][1]), float(rows[0][2])
# #     m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

# #     for row in rows:
# #         t, lat, lon, alt = row
# #         folium.Marker(
# #             location=[float(lat), float(lon)],
# #             popup=f"Survivor @ {t} (Alt: {alt})",
# #             icon=folium.Icon(color="red", icon="info-sign")
# #         ).add_to(m)

# #     m.save(MAP_FILE)
# #     print(f"🗺️ Map updated → {MAP_FILE}")


# # def log_survivor_detection(position):
# #     """Log survivor detection and update map + sound alert"""
# #     lat, lon, alt = position.latitude, position.longitude, position.altitude
# #     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

# #     # Save to CSV
# #     with open(CSV_FILE, "a", newline="") as f:
# #         writer = csv.writer(f)
# #         writer.writerow([timestamp, lat, lon, alt])

# #     # Print alert to console
# #     print(f"🚨 Survivor Detected @ {timestamp}")
# #     print(f"   Latitude: {lat:.6f}, Longitude: {lon:.6f}, Altitude: {alt:.2f} m")

# #     # Audio alert 🔊
# #     winsound.Beep(1000, 500)  # (frequency=1000Hz, duration=500ms)

# #     # Update map
# #     create_map()


# # # --- Obstacle Avoidance (Horizontal only, No Climb) ---
# # def obstacle_avoidance(client, depth_frame):
# #     h, w = depth_frame.shape
# #     left = np.mean(depth_frame[:, :w//3])
# #     center = np.mean(depth_frame[:, w//3:2*w//3])
# #     right = np.mean(depth_frame[:, 2*w//3:])

# #     DEPTH_HISTORY.append((left, center, right))

# #     avg_left = np.mean([d[0] for d in DEPTH_HISTORY])
# #     avg_center = np.mean([d[1] for d in DEPTH_HISTORY])
# #     avg_right = np.mean([d[2] for d in DEPTH_HISTORY])

# #     if avg_center < 5.0:  # Obstacle ahead
# #         if avg_left > avg_right:
# #             print("⚠️ Obstacle → Strafe Left")
# #             client.moveByVelocityBodyFrameAsync(2.0, -2.0, 0, 1).join()
# #         else:
# #             print("⚠️ Obstacle → Strafe Right")
# #             client.moveByVelocityBodyFrameAsync(2.0, 2.0, 0, 1).join()
# #     elif avg_center < 10.0:
# #         print("⚠️ Path partly blocked → Slowing Down")
# #         client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 1).join()
# #     else:
# #         client.moveByVelocityBodyFrameAsync(3.5, 0, 0, 1).join()  # Fast forward


# # # --- Main Drone Logic ---
# # def main():
# #     # Setup CSV if not exists
# #     if not os.path.exists(CSV_FILE):
# #         with open(CSV_FILE, "w", newline="") as f:
# #             writer = csv.writer(f)
# #             writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

# #     # Connect to AirSim
# #     client = airsim.MultirotorClient()
# #     client.confirmConnection()
# #     client.enableApiControl(True)
# #     client.armDisarm(True)
# #     print("🚁 Taking off...")
# #     client.takeoffAsync().join()

# #     # Initialize YOLOv5 detector
# #     detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

# #     try:
# #         while True:
# #             # Get scene + depth
# #             responses = client.simGetImages([
# #                 airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
# #                 airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
# #             ])
# #             if not responses or responses[0].height == 0:
# #                 time.sleep(0.1)
# #                 continue

# #             # RGB image
# #             img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
# #             frame = img1d.reshape(responses[0].height, responses[0].width, 3)
# #             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# #             # Depth image
# #             depth1d = np.array(responses[1].image_data_float, dtype=np.float32)
# #             depth_frame = depth1d.reshape(responses[1].height, responses[1].width)

# #             # --- Fast Obstacle Avoidance (No Climb) ---
# #             obstacle_avoidance(client, depth_frame)

# #             # --- YOLO Detection ---
# #             annotated, detections = detector.detect(frame_bgr)
# #             if len(detections) > 0:
# #                 pose = client.getGpsData().gnss.geo_point
# #                 log_survivor_detection(pose)

# #             # Show detection frame
# #             cv2.imshow("AirSim YOLOv5 + Obstacle Avoidance", annotated)
# #             if cv2.waitKey(1) & 0xFF == ord("q"):
# #                 break

# #     finally:
# #         print("🛑 Program exiting, drone will hover (not forced landing).")
# #         client.armDisarm(False)
# #         client.enableApiControl(False)
# #         cv2.destroyAllWindows()


# # if __name__ == "__main__":
# #     main()
##..
# # import airsim
# # import cv2
# # import numpy as np
# # import time
# # import csv
# # import folium
# # import os
# # import winsound
# # import threading
# # from collections import deque
# # from ai.yolov5_detector import YOLOv5PersonDetector


# # CSV_FILE = "survivor_detections.csv"
# # MAP_FILE = "survivor_map.html"

# # DEPTH_HISTORY = deque(maxlen=2)

# # # Shared globals
# # latest_frame = None
# # latest_detections = []
# # frame_lock = threading.Lock()
# # running = True


# # # ================= YOLO THREAD =================

# # def yolo_worker(detector):
# #     global latest_frame, latest_detections, running

# #     while running:
# #         if latest_frame is None:
# #             time.sleep(0.01)
# #             continue

# #         with frame_lock:
# #             frame_copy = latest_frame.copy()

# #         # Resize for speed (416x416)
# #         resized = cv2.resize(frame_copy, (416, 416))

# #         annotated, detections = detector.detect(resized)

# #         latest_detections = detections

# #         if len(detections) > 0:
# #             winsound.Beep(1000, 150)


# # # ================= FAST OBSTACLE =================

# # def obstacle_avoidance(client, depth_frame):

# #     h, w = depth_frame.shape
# #     band = depth_frame[h//3:2*h//3, :]

# #     left = float(np.min(band[:, :w//3]))
# #     center = float(np.min(band[:, w//3:2*w//3]))
# #     right = float(np.min(band[:, 2*w//3:]))

# #     DEPTH_HISTORY.append(center)
# #     avg_center = float(np.mean(DEPTH_HISTORY))

# #     base_speed = 7.0
# #     safety = 4.5

# #     if avg_center < safety:
# #         client.moveByVelocityBodyFrameAsync(0.0, 0.0, 0.0, 0.05)

# #         if left > right:
# #             client.rotateByYawRateAsync(-150.0, 0.25)
# #         else:
# #             client.rotateByYawRateAsync(150.0, 0.25)

# #         return

# #     client.moveByVelocityBodyFrameAsync(base_speed, 0.0, 0.0, 0.08)


# # # ================= MAIN =================

# # def main():

# #     global latest_frame, running

# #     client = airsim.MultirotorClient()
# #     client.confirmConnection()
# #     client.enableApiControl(True)
# #     client.armDisarm(True)

# #     print("🚁 Taking off...")
# #     client.takeoffAsync().join()

# #     detector = YOLOv5PersonDetector("yolov5s", 0.45)

# #     # Start YOLO thread
# #     yolo_thread = threading.Thread(target=yolo_worker, args=(detector,))
# #     yolo_thread.start()

# #     try:
# #         while True:

# #             responses = client.simGetImages([
# #                 airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
# #                 airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
# #             ])

# #             if not responses:
# #                 continue

# #             # RGB Frame
# #             img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
# #             frame = img1d.reshape(responses[0].height,
# #                                   responses[0].width, 3)
# #             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

# #             # Share frame safely with YOLO thread
# #             with frame_lock:
# #                 latest_frame = frame_bgr

# #             # Depth frame
# #             depth1d = np.array(responses[1].image_data_float, dtype=np.float32)
# #             depth_frame = depth1d.reshape(responses[1].height,
# #                                           responses[1].width)

# #             obstacle_avoidance(client, depth_frame)

# #             cv2.imshow("🚀 Ultra-Fast AI Drone", frame_bgr)

# #             if cv2.waitKey(1) & 0xFF == ord("q"):
# #                 break

# #     finally:
# #         running = False
# #         yolo_thread.join()

# #         client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.5)
# #         client.armDisarm(False)
# #         client.enableApiControl(False)
# #         cv2.destroyAllWindows()


# # if __name__ == "__main__":
# #     main()

# import airsim
# import cv2
# import numpy as np
# import time
# import csv
# import folium
# import os
# import winsound
# import threading
# from collections import deque
# from ai.yolov5_detector import YOLOv5PersonDetector


# CSV_FILE = "survivor_detections.csv"
# MAP_FILE = "survivor_map.html"

# DEPTH_HISTORY = deque(maxlen=2)

# latest_frame = None
# latest_detections = []
# frame_lock = threading.Lock()
# running = True


# # ================= MAP =================

# def create_map():
#     if not os.path.exists(CSV_FILE):
#         return

#     rows = []
#     with open(CSV_FILE, "r") as f:
#         reader = csv.reader(f)
#         next(reader, None)
#         rows = list(reader)

#     if not rows:
#         return

#     center_lat, center_lon = float(rows[0][1]), float(rows[0][2])
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

#     for row in rows:
#         t, lat, lon, alt = row
#         folium.Marker(
#             location=[float(lat), float(lon)],
#             popup=f"Survivor @ {t}",
#             icon=folium.Icon(color="red")
#         ).add_to(m)

#     m.save(MAP_FILE)


# def log_survivor_detection(client):
#     pose = client.getGpsData().gnss.geo_point

#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#     lat = float(pose.latitude)
#     lon = float(pose.longitude)
#     alt = float(pose.altitude)

#     with open(CSV_FILE, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([timestamp, lat, lon, alt])

#     print(f"🚨 Survivor detected @ {timestamp}")
#     winsound.Beep(1200, 300)
#     create_map()


# # ================= YOLO THREAD =================

# def yolo_worker(detector):
#     global latest_frame, latest_detections, running

#     while running:
#         if latest_frame is None:
#             time.sleep(0.01)
#             continue

#         with frame_lock:
#             frame_copy = latest_frame.copy()

#         # Resize for speed
#         resized = cv2.resize(frame_copy, (416, 416))

#         _, detections = detector.detect(resized)
#         latest_detections = detections


# # ================= OBSTACLE =================

# def obstacle_avoidance(client, depth_frame):

#     h, w = depth_frame.shape
#     band = depth_frame[h//3:2*h//3, :]

#     center = float(np.min(band[:, w//3:2*w//3]))
#     DEPTH_HISTORY.append(center)
#     avg_center = float(np.mean(DEPTH_HISTORY))

#     base_speed = 7.0
#     safety = 5.0

#     if avg_center < safety:
#         client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.05)
#         client.rotateByYawRateAsync(150, 0.25)
#         return

#     client.moveByVelocityBodyFrameAsync(base_speed, 0, 0, 0.08)


# # ================= MAIN =================

# def main():

#     global latest_frame, latest_detections, running

#     TARGET_ALTITUDE = -20  # Fly at 20 meters

#     if not os.path.exists(CSV_FILE):
#         with open(CSV_FILE, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

#     client = airsim.MultirotorClient()
#     client.confirmConnection()
#     client.enableApiControl(True)
#     client.armDisarm(True)

#     print("🚁 Taking off...")
#     client.takeoffAsync().join()

#     client.moveToZAsync(TARGET_ALTITUDE, 3).join()
#     print("✈ Flying at 20 meters")

#     detector = YOLOv5PersonDetector("yolov5s", 0.45)

#     yolo_thread = threading.Thread(target=yolo_worker, args=(detector,))
#     yolo_thread.start()

#     try:
#         while True:

#             responses = client.simGetImages([
#                 airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
#                 airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
#             ])

#             if not responses:
#                 continue

#             img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
#             frame = img1d.reshape(responses[0].height,
#                                   responses[0].width, 3)
#             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             with frame_lock:
#                 latest_frame = frame_bgr

#             # If detection found → log GPS
#             if len(latest_detections) > 0:
#                 log_survivor_detection(client)
#                 latest_detections = []

#             depth1d = np.array(responses[1].image_data_float, dtype=np.float32)
#             depth_frame = depth1d.reshape(responses[1].height,
#                                           responses[1].width)

#             obstacle_avoidance(client, depth_frame)

#             cv2.imshow("🚀 Fast AI Drone", frame_bgr)

#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     finally:
#         running = False
#         yolo_thread.join()

#         client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.5)
#         client.armDisarm(False)
#         client.enableApiControl(False)
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

##working but it is not marking persons
##...



# import airsim
# import cv2
# import numpy as np
# import time
# import csv
# import folium
# import os
# import winsound
# import threading
# from collections import deque
# from ai.yolov5_detector import YOLOv5PersonDetector


# CSV_FILE = "survivor_detections.csv"
# MAP_FILE = "survivor_map.html"

# DEPTH_HISTORY = deque(maxlen=2)

# latest_frame = None
# latest_detections = []
# frame_lock = threading.Lock()
# running = True


# # ================= MAP =================

# def create_map():
#     if not os.path.exists(CSV_FILE):
#         return

#     rows = []
#     with open(CSV_FILE, "r") as f:
#         reader = csv.reader(f)
#         next(reader, None)
#         rows = list(reader)

#     if not rows:
#         return

#     center_lat, center_lon = float(rows[0][1]), float(rows[0][2])
#     m = folium.Map(location=[center_lat, center_lon], zoom_start=18)

#     for row in rows:
#         t, lat, lon, alt = row
#         folium.Marker(
#             location=[float(lat), float(lon)],
#             popup=f"Survivor @ {t}",
#             icon=folium.Icon(color="red")
#         ).add_to(m)

#     m.save(MAP_FILE)


# def log_survivor_detection(client):
#     pose = client.getGpsData().gnss.geo_point

#     timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#     lat = float(pose.latitude)
#     lon = float(pose.longitude)
#     alt = float(pose.altitude)

#     with open(CSV_FILE, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([timestamp, lat, lon, alt])

#     print(f"🚨 Survivor detected @ {timestamp}")
#     winsound.Beep(1000, 200)
#     create_map()


# # ================= YOLO THREAD =================

# def yolo_worker(detector):
#     global latest_frame, latest_detections, running

#     while running:
#         if latest_frame is None:
#             time.sleep(0.01)
#             continue

#         with frame_lock:
#             frame_copy = latest_frame.copy()

#         resized = cv2.resize(frame_copy, (416, 416))
#         _, detections = detector.detect(resized)
#         latest_detections = detections


# # ================= OBSTACLE AVOIDANCE =================

# def obstacle_avoidance(client, depth_frame):

#     h, w = depth_frame.shape
#     band = depth_frame[h//3:2*h//3, :]

#     center = float(np.min(band[:, w//3:2*w//3]))
#     DEPTH_HISTORY.append(center)
#     avg_center = float(np.mean(DEPTH_HISTORY))

#     base_speed = 6.5
#     safety_distance = 5.0

#     if avg_center < safety_distance:
#         client.moveByVelocityBodyFrameAsync(0.0, 0.0, 0.0, 0.05)

#         left = float(np.min(band[:, :w//3]))
#         right = float(np.min(band[:, 2*w//3:]))

#         if left > right:
#             client.rotateByYawRateAsync(-120.0, 0.25)
#         else:
#             client.rotateByYawRateAsync(120.0, 0.25)

#         return

#     client.moveByVelocityBodyFrameAsync(base_speed, 0.0, 0.0, 0.1)


# # ================= MAIN =================

# def main():

#     global latest_frame, latest_detections, running

#     TARGET_ALTITUDE = -20  # 20 meters high

#     if not os.path.exists(CSV_FILE):
#         with open(CSV_FILE, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["Time", "Latitude", "Longitude", "Altitude (m)"])

#     client = airsim.MultirotorClient()
#     client.confirmConnection()
#     client.enableApiControl(True)
#     client.armDisarm(True)

#     print("🚁 Taking off...")
#     client.takeoffAsync().join()

#     # Fly higher
#     client.moveToZAsync(TARGET_ALTITUDE, 4).join()
#     print("✈ Flying at 20 meters")

#     # ✅ SAFE downward camera tilt (works on your AirSim version)
#     pitch = np.radians(-80)

#     camera_pose = airsim.Pose(
#         airsim.Vector3r(0, 0, 0),
#         airsim.to_quaternion(pitch, 0, 0)
#     )

#     client.simSetCameraPose("0", camera_pose)
#     print("📷 Camera tilted downward")

#     detector = YOLOv5PersonDetector("yolov5s", 0.40)

#     yolo_thread = threading.Thread(target=yolo_worker, args=(detector,))
#     yolo_thread.start()

#     try:
#         while True:

#             responses = client.simGetImages([
#                 airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
#                 airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
#             ])

#             if not responses:
#                 continue

#             img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
#             frame = img1d.reshape(responses[0].height,
#                                   responses[0].width, 3)
#             frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#             with frame_lock:
#                 latest_frame = frame_bgr

#             if len(latest_detections) > 0:
#                 log_survivor_detection(client)
#                 latest_detections = []

#             depth1d = np.array(responses[1].image_data_float,
#                                dtype=np.float32)

#             depth_frame = depth1d.reshape(responses[1].height,
#                                           responses[1].width)

#             obstacle_avoidance(client, depth_frame)

#             cv2.imshow("🚀 AI DRONE - DOWNWARD VIEW", frame_bgr)

#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#     finally:
#         running = False
#         yolo_thread.join()

#         client.moveByVelocityBodyFrameAsync(0.0, 0.0, 0.0, 0.5)
#         client.armDisarm(False)
#         client.enableApiControl(False)
#         cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

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

print("🚀 Script Started Successfully")

# ================= CONFIG =================

CSV_FILE = "survivor_detections.csv"
MAP_FILE = "survivor_map.html"
TARGET_ALTITUDE = -20
DETECTION_COOLDOWN = 5

DEPTH_HISTORY = deque(maxlen=2)

latest_frame = None
latest_detections = []
frame_lock = threading.Lock()
running = True
last_detection_time = 0

# ================= MAP =================

def create_map():
    if not os.path.exists(CSV_FILE):
        return

    rows = []
    with open(CSV_FILE, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = list(reader)

    if not rows:
        return

    center_lat, center_lon = float(rows[-1][1]), float(rows[-1][2])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=19)

    for row in rows:
        t, lat, lon, alt = row
        folium.Marker(
            location=[float(lat), float(lon)],
            popup=f"🆘 Survivor @ {t}",
            icon=folium.Icon(color="red")
        ).add_to(m)

    m.save(MAP_FILE)
    print("🗺 Map Updated")

# ================= LOGGING =================

def log_survivor_detection(client):
    global last_detection_time

    current_time = time.time()
    if current_time - last_detection_time < DETECTION_COOLDOWN:
        return

    last_detection_time = current_time

    pose = client.getGpsData().gnss.geo_point
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    lat = float(pose.latitude)
    lon = float(pose.longitude)
    alt = float(pose.altitude)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, lat, lon, alt])

    print(f"🚨 Survivor Logged @ {timestamp}")
    winsound.Beep(1200, 300)

    create_map()

# ================= YOLO THREAD =================

def yolo_worker(detector):
    global latest_frame, latest_detections, running

    print("🧠 YOLO Thread Started")

    while running:
        try:
            if latest_frame is None:
                time.sleep(0.01)
                continue

            with frame_lock:
                frame_copy = latest_frame.copy()

            resized = cv2.resize(frame_copy, (416, 416))
            _, detections = detector.detect(resized)
            latest_detections = detections

        except Exception as e:
            print("YOLO ERROR:", e)

# ================= OBSTACLE =================

def obstacle_avoidance(client, depth_frame):
    h, w = depth_frame.shape
    band = depth_frame[h//3:2*h//3, :]

    center = float(np.min(band[:, w//3:2*w//3]))
    DEPTH_HISTORY.append(center)
    avg_center = float(np.mean(DEPTH_HISTORY))

    speed = 6.0
    safety_distance = 5.0

    if avg_center < safety_distance:
        client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.05)

        left = float(np.min(band[:, :w//3]))
        right = float(np.min(band[:, 2*w//3:]))

        if left > right:
            client.rotateByYawRateAsync(-120, 0.25)
        else:
            client.rotateByYawRateAsync(120, 0.25)

        return

    client.moveByVelocityBodyFrameAsync(speed, 0, 0, 0.1)

# ================= MAIN =================

def main():
    global latest_frame, latest_detections, running

    print("🔗 Connecting to AirSim...")

    # CSV header setup
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Latitude", "Longitude", "Altitude"])

    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("🚁 Taking off...")
    client.takeoffAsync().join()

    client.moveToZAsync(TARGET_ALTITUDE, 4).join()
    print("✈ Flying at 20 meters")

    # Downward camera tilt
    pitch = np.radians(-80)
    camera_pose = airsim.Pose(
        airsim.Vector3r(0, 0, 0),
        airsim.to_quaternion(pitch, 0, 0)
    )
    client.simSetCameraPose("0", camera_pose)
    print("📷 Camera Tilted Down")

    detector = YOLOv5PersonDetector("yolov5s", 0.40)

    yolo_thread = threading.Thread(target=yolo_worker, args=(detector,))
    yolo_thread.start()

    try:
        while True:
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
                airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
            ])

            if not responses:
                continue

            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            frame = img1d.reshape(responses[0].height,
                                  responses[0].width, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            with frame_lock:
                latest_frame = frame_bgr

            if len(latest_detections) > 0:
                log_survivor_detection(client)
                latest_detections = []

            depth1d = np.array(responses[1].image_data_float, dtype=np.float32)
            depth_frame = depth1d.reshape(responses[1].height,
                                          responses[1].width)

            obstacle_avoidance(client, depth_frame)

            cv2.imshow("🚀 AI DRONE", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("🛑 Interrupted")

    finally:
        running = False
        time.sleep(0.5)

        try:
            client.armDisarm(False)
            client.enableApiControl(False)
        except:
            pass

        cv2.destroyAllWindows()

# ================= EXECUTION =================

if __name__ == "__main__":
    main()
