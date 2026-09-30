# 🚁 AI Drone Search & Rescue

### *Autonomous Skies | AI Vision | Real-Time Rescue*

> **“Eyes in the sky, intelligence in flight — detecting, mapping, and assisting rescue operations with every second.”**

---

## 🌟 Project Overview

**AI Drone Search & Rescue** is an **AI-powered autonomous drone search-and-rescue simulation system** designed to detect people from a drone's camera, identify their location, avoid obstacles, and record detection information automatically.

The project combines:

* 🚁 **AirSim** — realistic drone flight simulation
* 👁️ **YOLOv5** — real-time person/survivor detection
* 📷 **OpenCV** — camera and image processing
* 🚧 **Depth Camera** — obstacle detection and avoidance
* 🗺️ **Folium** — interactive survivor-location mapping
* 📊 **CSV** — detection and telemetry logging
* 🔊 **Audio/Visual Alerts** — immediate survivor detection feedback

The system is designed as a simulation platform that can later be extended toward real-world autonomous drone applications.

---

## 🎯 Objectives

The main objectives of this project are:

1. Detect survivors/persons automatically using computer vision.
2. Navigate a simulated drone using AirSim.
3. Detect and avoid obstacles during flight.
4. Record the detected person's location and flight information.
5. Generate an interactive map showing detected survivor locations.
6. Provide real-time visual and audio alerts.
7. Create a modular architecture that can be extended to real drone platforms.

---

## 🚀 Key Features

### 1. 👁️ Real-Time Survivor Detection

The drone camera continuously captures images while the YOLOv5 model searches for people.

When a person is detected, the system can record:

* Detection time
* GPS coordinates
* Altitude
* Detection information
* Confidence score

---

### 2. 🚧 Intelligent Obstacle Avoidance

The system uses AirSim's depth-camera data to identify obstacles in the drone's flight path.

The obstacle-avoidance module can:

* Detect nearby obstacles
* Estimate obstacle distance
* Change the drone's movement direction
* Continue searching after avoiding obstacles

---

### 3. 🗺️ Automatic Location Mapping

Detected survivor locations are stored and visualized using **Folium**.

The generated map can be opened in a web browser to view the detected locations.

Example output:

```text
survivor_map.html
```

---

### 4. 📊 Automatic Detection Logging

Detection information is saved in CSV format.

Example:

```text
survivor_detections.csv
```

This makes it possible to analyze:

* Detection time
* Location
* Altitude
* Number of detections
* Other telemetry information

---

### 5. 📹 Real-Time Camera Processing

OpenCV is used to process the drone camera feed and display detection results.

The system can display:

* Live camera frames
* Bounding boxes
* Person labels
* Confidence scores
* Detection information

Press:

```text
Q
```

to exit the camera window when supported by the running script.

---

### 6. 🔊 Audio & Visual Alerts

The system provides immediate feedback when a person is detected through visual annotations and detection alerts.

---

### 7. 🧩 Modular Architecture

The project is organized into separate components so individual modules can be modified or replaced.

For example:

```text
YOLOv5
   ↓
YOLO Detector
   ↓
Drone Camera
   ↓
Person Detection
   ↓
GPS / Telemetry
   ↓
CSV Logging
   ↓
Folium Map
```

The detection model can also be replaced with another compatible object-detection model in the future.

---

# 🏗️ System Architecture

```text
                  ┌─────────────────────┐
                  │       AirSim        │
                  │  Drone Simulation   │
                  └──────────┬──────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │    Drone Camera     │
                  │   RGB / Depth       │
                  └──────────┬──────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
        ┌─────────────────┐   ┌──────────────────┐
        │    YOLOv5       │   │ Depth Processing │
        │ Person Detection│   │ Obstacle Detect. │
        └────────┬────────┘   └────────┬─────────┘
                 │                     │
                 ▼                     ▼
        ┌─────────────────┐   ┌──────────────────┐
        │ Survivor        │   │ Obstacle         │
        │ Detection       │   │ Avoidance        │
        └────────┬────────┘   └────────┬─────────┘
                 │                     │
                 └──────────┬──────────┘
                            ▼
                  ┌─────────────────────┐
                  │ GPS / Telemetry     │
                  └──────────┬──────────┘
                             │
                    ┌────────┴────────┐
                    ▼                 ▼
          ┌─────────────────┐ ┌─────────────────┐
          │ CSV Detection   │ │ Folium Map      │
          │ Logging         │ │ Visualization   │
          └─────────────────┘ └─────────────────┘
```

---

# 📁 Project Structure

```text
AI_Drone_SearchRescue/
│
├── ai/
│   └── YOLO detection modules
│
├── drone/
│   └── Drone control and AirSim modules
│
├── dashboard/
│   └── Dashboard / visualization components
│
├── yolov5/
│   └── YOLOv5 model and related files
│
├── AirSimEnvironments/
│   └── AirSim simulation environments
│
├── main_airsim_survivor.py
│
├── main_airsim_survivor_obstacle.py
│
├── main_airsim_obstacle_yolo.py
│
├── obstacle_dodge.py
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

> The exact files available may vary depending on the current project version.

---

# ⚡ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/kafkakaif/AI_Drone_SearchRescue.git
```

Move into the project directory:

```bash
cd AI_Drone_SearchRescue
```

---

## 2. Create a Virtual Environment

It is recommended to use a separate Python virtual environment.

### Windows

```bash
python -m venv airsim_env
```

Activate it using PowerShell:

```bash
.\airsim_env\Scripts\Activate.ps1
```

Or Command Prompt:

```bash
airsim_env\Scripts\activate.bat
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Python Dependencies

The project includes a `requirements.txt` file.

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

This installs the main Python dependencies required by the project, including:

* AirSim Python API
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Pandas
* Folium
* Matplotlib
* Seaborn
* SciPy
* PyYAML
* tqdm
* Requests
* Pillow

---

# 🖥️ AirSim Setup

**Important:** Installing `requirements.txt` installs the Python dependencies, but it does **not** install the AirSim simulator/environment itself.

You must have AirSim configured separately.

Supported environments used by this project may include:

```text
Blocks
CityEnviron
```

Place/configure the required AirSim environments according to your AirSim installation.

---

# 🤖 YOLOv5 Model

The project uses **YOLOv5** for person/survivor detection.

Make sure the required model weights are available before running the detection scripts.

Typical model files may include:

```text
best.pt
yolov5s.pt
```

The exact model depends on the implementation and trained model being used.

---

# ▶️ Running the Project

First make sure:

1. Your virtual environment is activated.
2. AirSim is installed and configured.
3. Your AirSim environment is running.
4. Required YOLO model weights are available.
5. Python dependencies are installed.

Then run the required script.

### Survivor Detection + Obstacle Avoidance

```bash
python main_airsim_survivor_obstacle.py
```

### Survivor Detection

```bash
python main_airsim_survivor.py
```

### YOLO + Obstacle Detection

```bash
python main_airsim_obstacle_yolo.py
```

---

# 📊 Output

After running the system, detection information can be stored in:

```text
survivor_detections.csv
```

The interactive map can be generated as:

```text
survivor_map.html
```

Open the HTML file in a browser to visualize detected survivor locations.

---

# 🧪 Example Workflow

```text
Start AirSim
     ↓
Launch Python Detection Script
     ↓
Drone Takes Off
     ↓
Camera Captures Frames
     ↓
YOLOv5 Detects Person
     ↓
Survivor Detected
     ↓
GPS / Telemetry Retrieved
     ↓
Detection Saved to CSV
     ↓
Location Added to Folium Map
     ↓
Drone Continues Search
     ↓
Obstacle Detected?
     ↓
Yes → Avoid Obstacle
     ↓
Continue Search
```

---

# 🛠️ Technologies Used

| Technology    | Purpose                    |
| ------------- | -------------------------- |
| Python        | Main programming language  |
| AirSim        | Drone simulation           |
| Unreal Engine | Simulation environment     |
| YOLOv5        | Person/survivor detection  |
| PyTorch       | Deep learning framework    |
| OpenCV        | Image and video processing |
| NumPy         | Numerical processing       |
| Pandas        | Data processing            |
| Folium        | Interactive maps           |
| Matplotlib    | Visualization              |
| Seaborn       | Data visualization         |
| CSV           | Detection logging          |

---

# 💻 System Requirements

### Recommended

```text
OS              : Windows 10 / 11
Python          : 3.10+
RAM             : 8 GB minimum
GPU             : NVIDIA GPU recommended
VRAM            : 4 GB+ recommended
Storage         : 10 GB+ free space
```

An NVIDIA GPU is recommended for faster YOLO inference and simulation performance.

---

# ⚠️ Important Notes

### AirSim

AirSim is a simulator and must be running before the Python client can communicate with the drone.

### GPU / CUDA

PyTorch installation can differ depending on your NVIDIA GPU and CUDA configuration.

If you specifically want GPU acceleration, install a PyTorch build compatible with your system.

You can check CUDA availability with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns:

```text
True
```

PyTorch can access your NVIDIA GPU.

---

# 🔧 Troubleshooting

### `ModuleNotFoundError`

Example:

```text
ModuleNotFoundError: No module named 'airsim'
```

Run:

```bash
pip install -r requirements.txt
```

and make sure your virtual environment is activated.

---

### AirSim Connection Error

Make sure the AirSim environment is running before executing the Python script.

---

### CUDA Not Available

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If it returns `False`, verify your NVIDIA driver and PyTorch installation.

---


# 📜 License

This project is intended for educational, research, and demonstration purposes.

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

**AI Vision. Autonomous Flight. Smarter Search & Rescue. 🚁**
