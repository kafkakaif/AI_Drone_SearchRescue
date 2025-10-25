# 🚁 DroneSurvivorDetection 🌌
*Autonomous Skies | AI Vision | Real-Time Rescue*

---

> **“Eyes in the sky, intelligence in flight — mapping, detecting, and saving lives with every second.”**

![Drone Hero](https://images.unsplash.com/photo-1529921879218-1c4f9d9e70f1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8ZHJvbmV8fHx8fDE2ODk4MjY3NjI&ixlib=rb-4.0.3&q=80&w=1080)

---

## 🌟 Project Overview

**DroneSurvivorDetection** is an **AI-powered autonomous drone system** designed for **search & rescue simulations**.  
Using **AirSim** for hyper-realistic flight simulation and **YOLOv5** for person detection, this project enables drones to:

- Detect survivors in real-time  
- Avoid obstacles intelligently  
- Log and map positions automatically  

> **“From virtual skies to real-world missions — intelligent drones at your service.”**

---

## 🚀 Key Features

### **1. Real-Time Survivor Detection**
- Fast and precise person detection using YOLOv5  
- Logs GPS coordinates, altitude, and timestamp for each detection  

### **2. Intelligent Obstacle Avoidance**
- Depth camera-based obstacle detection  
- Smooth horizontal avoidance ensures safe flight paths  

### **3. Automatic Logging & Mapping**
- CSV logging for all detections  
- Live interactive maps generated using **Folium**  

### **4. Simulation-Ready**
- Compatible with AirSim environments like **Blocks** & **CityEnviron**  
- Easily adaptable for **real drones** (PX4, DJI SDK, ESP32)  

### **5. Audio & Visual Alerts**
- Beep alerts on detection  
- Annotated live video feed for instant feedback  

### **6. Modular & Extensible**
- YOLOv5 detector can be replaced with YOLOv8 or other models  
- Obstacle avoidance, logging, and mapping are modular  

---

## ⚡ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/kafkakaif/DroneSurvivorDetection.git
cd DroneSurvivorDetection
Create a virtual environment

bash
Copy code
python -m venv airsim_env
Activate the environment

Windows (PowerShell):

powershell
Copy code
.\airsim_env\Scripts\Activate.ps1
Windows (CMD):

cmd
Copy code
.\airsim_env\Scripts\activate.bat
Linux / macOS:

bash
Copy code
source airsim_env/bin/activate
Install dependencies

bash
Copy code
pip install -r requirements.txt
Place AirSim environments

Put your maps (CityEnviron / Blocks) in AirSimEnvironments/

🖥️ Usage
Run the main script:

bash
Copy code
python main_airsim_survivor_obstacle.py
Press q to quit simulation

Detected survivors logged in survivor_detections.csv

Map automatically generated in survivor_map.html

📦 Requirements
Python 3.10+

AirSim

PyTorch

OpenCV

YOLOv5

Folium

Seaborn, Pandas, tqdm