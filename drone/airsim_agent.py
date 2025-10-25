import airsim
import numpy as np
import cv2
import time

class AirSimAgent:
    """
    Thin wrapper around AirSim's MultirotorClient to:
    - connect, arm, takeoff/land
    - grab RGB & Depth images
    - issue simple forward/turn/hover commands
    """

    def __init__(self, ip: str = "127.0.0.1"):
        self.client = airsim.MultirotorClient(ip)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

    def takeoff(self, altitude: float = -3.0):
        # AirSim uses NED frame; negative Z is up.
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(altitude, 1.0).join()

    def land(self):
        try:
            self.client.landAsync().join()
        finally:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)

    def get_rgb_depth(self, width: int = 640, height: int = 360):
        # Request Scene (RGB) and DepthPerspective (float meters)
        requests = [
            airsim.ImageRequest("0", airsim.ImageType.Scene, pixels_as_float=False, compress=False),
            airsim.ImageRequest("0", airsim.ImageType.DepthPerspective, pixels_as_float=True, compress=False),
        ]
        responses = self.client.simGetImages(requests)

        # RGB
        rgb_resp = responses[0]
        rgb = np.frombuffer(rgb_resp.image_data_uint8, dtype=np.uint8)
        rgb = rgb.reshape(rgb_resp.height, rgb_resp.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Depth (float32 meters)
        depth_resp = responses[1]
        depth = np.array(depth_resp.image_data_float, dtype=np.float32)
        depth = depth.reshape(depth_resp.height, depth_resp.width)

        # Resize to common size
        bgr = cv2.resize(bgr, (width, height))
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
        return bgr, depth

    def move_forward(self, vel: float = 2.0, duration: float = 0.5):
        # Body frame forward is +X
        self.client.moveByVelocityBodyFrameAsync(
            vx=vel, vy=0, vz=0, duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
        ).join()

    def turn(self, yaw_rate_deg: float = 30.0, duration: float = 1.0):
        self.client.moveByVelocityBodyFrameAsync(
            vx=0, vy=0, vz=0, duration=duration,
            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
        ).join()

    def hover(self, duration: float = 2.0):
        self.client.hoverAsync().join()
        time.sleep(max(0.0, duration))

    def get_pose(self):
        state = self.client.getMultirotorState()
        p = state.kinematics_estimated.position
        return float(p.x_val), float(p.y_val), float(p.z_val)
