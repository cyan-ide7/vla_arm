# Niryo VLA Robotic Arm Control System

This project implements a **Vision-Language-Action (VLA)** control system for the Niryo robotic arm in CoppeliaSim. It utilizes a hybrid architecture that combines local Large Language Models (**Ollama/Llama 3**), real-time object detection (**YOLOv8**), and **CNN-based Behavioral Cloning** for autonomous pick-and-place tasks.

## 🚀 System Architecture

The system operates through a decoupled multi-layer pipeline:
1.  **Language Layer**: Uses **Ollama (Llama 3)** to parse natural language instructions into structured goals.
2.  **Vision Layer**: Employs **YOLOv8** for object detection and a **3D Pose Resolver** (pinhole camera model) to determine world coordinates.
3.  **Action Layer**: A **CNN-based (ResNet-18)** Behavioral Cloning network predicts smooth trajectories based on visual feedback and current robot state.
4.  **Execution Layer**: CoppeliaSim's **Inverse Kinematics (IK)** solver translates predicted poses into physical joint commands.

## 🛠️ Prerequisites

Before installation, ensure you have the following:
* **Operating System**: Windows 10/11 or Ubuntu Linux.
* **CoppeliaSim**: Download and install [CoppeliaSim](https://www.coppeliarobotics.com/).
* **Ollama**: Install [Ollama](https://ollama.com/) and pull the Llama 3 model:
    ```bash
    ollama pull llama3
    ```
* **Python**: Version 3.8 or higher.

## 📥 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/cyan-ide7/vla_arm.git
    cd vla_arm
    ```

2.  **Create a Virtual Environment**:
    ```bash
    python -m venv vla_env
    # Windows:
    .\\vla_env\\Scripts\\Activate.ps1
    # Linux:
    source vla_env/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install ultralytics opencv-python numpy tqdm coppeliasim-zmqremoteapi_client requests
    ```

## 🎮 How to Run

### 1. Setup Simulation
* Open **CoppeliaSim**.
* Load your scene containing the **NiryoOne** arm.
* Ensure the **ZMQ Remote API server** is running (check `Add-ons` -> `ZMQ remote API server`).
* Start the simulation.

### 2. Execution Flow
Run the scripts in the following order:

* **Test Vision (Optional)**: Verify that YOLO and color masking are working correctly.
    ```bash
    python vision_parser.py
    ```
* **Inference (Live Execution)**: Run the main control loop to issue commands.
    ```bash
    python vla_interface.py
    ```
    *When prompted, enter a command like: "place the red cube on the white circle".*

## 📁 Project Structure

* `vla_interface.py`: The live inference loop connecting Ollama, YOLO, and the BC network.
* `vla_model.py`: NiryoVLA neural network definition (CNN + Fusion MLP).
* `vision_parser.py`: YOLOv8 detection and 2D-to-3D coordinate resolver.
* `action_config.py`: Defines workspace bounds and normalization parameters.
* `train_vla.py`: Training script for behavioral cloning.

## ⚠️ Common Troubleshooting
* **Connection Refused**: Ensure the ZMQ Remote API is enabled in CoppeliaSim and the simulation is active.
* **Ollama Timeout**: Ensure the Ollama service is running in the background before starting the interface.
* **Reach Error**: If the arm does not touch the object, adjust `Z_PICK_HEIGHT` in the control script.

---
**Admin/Contributor**: mars.ciot@pes.edu
