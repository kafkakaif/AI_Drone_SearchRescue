import airsim
import cv2
import numpy as np
import time

# ============================
# USER SETTINGS
# ============================
DRONE_SPEED = 5     # Forward speed in m/s (increase/decrease here)
DODGE_SPEED = 4     # Sideways dodge speed in m/s
SAFE_DISTANCE = 5.0 # Minimum distance before dodging obstacle
STEP_TIME = 2       # Duration of each movement command (seconds)

# ============================
# CONNECT TO AIRSIM
# ============================
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()
time.sleep(2)

print("Drone is flying with speed =", DRONE_SPEED)

# ============================
# MAIN LOOP
# ============================
try:
    while True:
        # Get depth image from front camera
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)
        ])

        # Convert depth to numpy array
        img1d = np.array(responses[0].image_data_float, dtype=np.float32)
        img2d = img1d.reshape(responses[0].height, responses[0].width)

        # Take center region (focus on what's ahead)
        center_region = img2d[100:200, 200:400]
        min_distance = np.min(center_region)

        print("Obstacle distance ahead:", round(min_distance, 2))

        # If obstacle too close → dodge
        if min_distance < SAFE_DISTANCE:
            print("⚠️ Obstacle detected! Dodging left...")
            client.moveByVelocityAsync(0, -DODGE_SPEED, 0, STEP_TIME).join()
        else:
            # Move forward at set speed
            client.moveByVelocityAsync(DRONE_SPEED, 0, 0, STEP_TIME).join()

except KeyboardInterrupt:
    print("Landing...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Drone stopped.")
