import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("✅ Connected to AirSim successfully!")

client.takeoffAsync().join()
print("🚁 Drone took off")
client.hoverAsync().join()
print("🛑 Drone hovering")
client.landAsync().join()
print("🏁 Drone landed safely")

client.armDisarm(False)
client.enableApiControl(False)
