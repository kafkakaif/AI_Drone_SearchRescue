import airsim
import numpy as np
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# Capture RGB + Depth
responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),        # RGB
    airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, True)      # Depth
])

# -------------------------------
# RGB image
rgb_img = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
rgb_img = rgb_img.reshape(responses[0].height, responses[0].width, 3)

# Depth image
depth_img = np.array(responses[1].image_data_float, dtype=np.float32)
depth_img = depth_img.reshape(responses[1].height, responses[1].width)

print("RGB shape:", rgb_img.shape)
print("Depth shape:", depth_img.shape)
print("Depth min/max (meters):", np.min(depth_img), np.max(depth_img))