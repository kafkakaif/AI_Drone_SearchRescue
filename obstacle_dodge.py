import airsim
import cv2
import numpy as np
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Takeoff
client.takeoffAsync().join()

def get_lidar_points():
    lidarData = client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
    if len(lidarData.point_cloud) < 3:
        return []
    points = np.array(lidarData.point_cloud, dtype=np.float32).reshape(-1, 3)
    return points

def detect_obstacle(points, threshold=3.0):
    """Check if obstacle within threshold distance ahead"""
    if len(points) == 0:
        return False
    # Check obstacles in front (X positive)
    front = points[(points[:,0] > 0) & (np.abs(points[:,1]) < 2)]
    if len(front) == 0:
        return False
    min_dist = np.min(np.linalg.norm(front, axis=1))
    return min_dist < threshold

while True:
    # Get RGB and Depth
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),
        airsim.ImageRequest("thermal", airsim.ImageType.Infrared, False, False)
    ])

    # RGB
    rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8).reshape(responses[0].height, responses[0].width, 3)

    # Depth (normalize for display)
    depth = np.array(responses[1].image_data_float, dtype=np.float32)
    depth = depth.reshape(responses[1].height, responses[1].width)
    depth_img = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Thermal
    thermal = np.frombuffer(responses[2].image_data_uint8, dtype=np.uint8).reshape(responses[2].height, responses[2].width, 3)

    # Lidar obstacle check
    points = get_lidar_points()
    obstacle = detect_obstacle(points)

    if obstacle:
        print("⚠️ Obstacle detected! Dodging...")
        client.moveByVelocityAsync(0, -2, 0, 2).join()  # Move left
    else:
        client.moveByVelocityAsync(2, 0, 0, 2).join()   # Move forward

    # Show feeds
    cv2.imshow("RGB", rgb)
    cv2.imshow("Depth", depth_img)
    cv2.imshow("Thermal", thermal)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Land
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()
