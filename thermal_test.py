import airsim
import cv2
import numpy as np

client = airsim.MultirotorClient()
client.confirmConnection()

# Get thermal image
responses = client.simGetImages([
    airsim.ImageRequest("thermal", airsim.ImageType.Infrared, False, False)
])

img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8) 
img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

cv2.imshow("Thermal Camera", img_rgb)
cv2.waitKey(0)
