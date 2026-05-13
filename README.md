# VLA-ARM: Vision-Language-Action for NiryoOne Robot Arm

This project implements a Vision-Language-Action (VLA) system for controlling a NiryoOne robot arm in a CoppeliaSim simulation environment. The system learns from human demonstrations to perform manipulation tasks based on natural language instructions.

## Project Overview

The VLA system consists of:
- **Vision Encoder**: Processes camera images from top and wrist viewpoints
- **Language Encoder**: Understands natural language task instructions
- **Action Decoder**: Predicts robot actions (6D pose deltas + gripper control)
- **State Integration**: Incorporates current robot state for context-aware predictions

## Prerequisites

- Python 3.10+
- CoppeliaSim (with ZMQ Remote API enabled)
- CUDA-compatible GPU (recommended for training)

## Installation

1. **Clone or navigate to the repository:**
   ```bash
   cd c:\Users\cyani\Desktop\vla_arm
   ```

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv vla_env
   & .\vla_env\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Ensure CoppeliaSim is running:**
   - Open CoppeliaSim
   - Load the scene: `vla_rm.ttt`
   - Start the simulation

## Usage

Follow these steps in order to set up and run the VLA system:

### 1. Record Demonstrations

Collect human demonstrations by teleoperating the robot in the simulation.

```powershell
python python/record.py
```

- Enter a natural language task instruction when prompted
- Teleoperate the robot using your keyboard/mouse in CoppeliaSim
- The system will record:
  - Camera images (top and wrist views)
  - Robot states (joint positions, end-effector pose, gripper state)
  - Actions taken
- Demonstrations are saved in the `dataset/` directory

**Note:** Record multiple demonstrations with varied instructions and scenarios for better training.

### 2. Preprocess Dataset

Process the raw demonstrations into a format suitable for training.

```powershell
python python/pre_process.py
```

This script will:
- Normalize robot states
- Generate action sequences from state differences
- Resize and preprocess images
- Save processed data to `processed_dataset/`

### 3. Train VLA Model

Train the Vision-Language-Action model using the processed dataset.

```powershell
python python/train_vla.py
```

Training parameters (configurable in the script):
- Batch size: 16
- Epochs: 20
- Learning rate: 1e-4
- Validation split: 10%

The trained model will be saved as `models/niryo_vla.pth`.

**Note:** Training requires a GPU for reasonable performance. Adjust batch size if encountering memory issues.

### 4. Run VLA Interface

Use the trained model to control the robot with natural language instructions.

```powershell
python python/vla_interface.py
```

- Enter task instructions (e.g., "Pick up the red cube and place it on the blue platform")
- The system will:
  - Capture current camera images and robot state
  - Generate action predictions using the VLA model
  - Execute actions in the CoppeliaSim simulation
  - Provide real-time feedback

## Project Structure

```
vla_arm/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── vla_rm.ttt               # CoppeliaSim scene file
├── dataset/                  # Raw demonstration data
├── processed_dataset/        # Preprocessed training data
├── models/                   # Trained model checkpoints
│   └── niryo_vla.pth        # Pre-trained VLA model
└── python/                   # Source code
    ├── record.py            # Demonstration recording
    ├── pre_process.py       # Data preprocessing
    ├── train_vla.py         # Model training
    ├── vla_interface.py     # Inference interface
    ├── vla_model.py         # VLA model architecture
    ├── action_config.py     # Action space configuration
    ├── image_ob.py          # Image observation utilities
  s
```

## Configuration

Key parameters can be adjusted in the respective scripts:

- **Action Space**: Modify `ACTION_LOW`, `ACTION_HIGH` in `action_config.py`
- **Model Architecture**: Update `vla_model.py` for different network designs
- **Training Hyperparameters**: Change batch size, epochs, learning rate in `train_vla.py`
- **Control Frequency**: Adjust `CONTROL_FREQUENCY` in `vla_interface.py`


