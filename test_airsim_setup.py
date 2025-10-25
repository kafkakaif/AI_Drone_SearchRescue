import airsim
import cv2
import numpy as np
from ai.yolov5_detector import YOLOv5PersonDetector

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
print("✅ Connected to AirSim")

# Test takeoff, hover, and landing
client.takeoffAsync().join()
print("🚁 Drone took off")
client.hoverAsync().join()
print("🛑 Drone hovering")
client.landAsync().join()
print("🏁 Drone landed safely")

# Test forward movement
client.takeoffAsync().join()
client.moveByVelocityBodyFrameAsync(1.5, 0, 0, 3).join()  # Forward for 3 seconds
client.hoverAsync().join()
client.landAsync().join()

# Test YOLOv5 detection
client.takeoffAsync().join()
detector = YOLOv5PersonDetector(model_name="yolov5s", conf_thres=0.45)

responses = client.simGetImages([
    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
])
img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
frame = img1d.reshape(responses[0].height, responses[0].width, 3)
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

annotated, detections = detector.detect(frame_bgr)
print("Detections:", detections)

cv2.imshow("YOLOv5 Test", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
