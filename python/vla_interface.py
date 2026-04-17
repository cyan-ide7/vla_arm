# python/vla_interface.py
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import torch
import numpy as np
import cv2
import time
import os
from torchvision import transforms
from PIL import Image
import open_clip
import warnings

from vla_model import NiryoVLA
from action_config import (
    ACTION_DIM,
    denormalize_action,
    clip_action
)

warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
STATE_DIM = 13
MODEL_PATH = os.path.join("models", "niryo_vla.pth")
CONTROL_FREQUENCY = 10  # Hz
SMOOTHING_FACTOR = 0.7  # For smoother robot motion

# --------------------------------------------------
# DEVICE SETUP
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = NiryoVLA(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("✅ Trained VLA model loaded.")

# --------------------------------------------------
# TOKENIZER FOR TEXT INSTRUCTIONS
# --------------------------------------------------
tokenizer = open_clip.get_tokenizer("ViT-B-32")

instruction = input(
    "Enter command (e.g., 'pick the cube and place it on the left'): "
)
text_tokens = tokenizer([instruction]).to(device)

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def convert_image(img_buffer, resolution):
    """Convert CoppeliaSim image buffer to OpenCV format."""
    if img_buffer is None or len(img_buffer) == 0:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    width, height = resolution
    img = np.frombuffer(img_buffer, dtype=np.uint8)
    img = img.reshape(height, width, 3)
    img = np.flipud(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

# --------------------------------------------------
# CONNECT TO COPPELIASIM
# --------------------------------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')

niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

print("✅ Connected to CoppeliaSim.")

# --------------------------------------------------
# OBSERVATION FUNCTION
# --------------------------------------------------
def get_observation():
    obs = sim.callScriptFunction('getObservation', script_handle, [])

    top_img = convert_image(obs['top_image'], obs['top_resolution'])
    wrist_img = convert_image(obs['wrist_image'], obs['wrist_resolution'])

    top_tensor = transform(Image.fromarray(top_img)).unsqueeze(0).to(device)
    wrist_tensor = transform(Image.fromarray(wrist_img)).unsqueeze(0).to(device)

    state_vector = np.array(
        obs['joint_positions']
        + obs['ee_position']
        + obs['ee_orientation']
        + [obs['gripper_state']],
        dtype=np.float32
    )
    state_tensor = torch.tensor(state_vector).unsqueeze(0).to(device)

    return top_tensor, wrist_tensor, state_tensor

# --------------------------------------------------
# APPLY ACTION
# --------------------------------------------------
prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

def apply_action(action_normalized):
    """
    Apply a normalized action to the robot.
    Smoothing is performed in normalized space.
    """
    global prev_action

    # Smooth the action
    action_smoothed = (
        SMOOTHING_FACTOR * prev_action
        + (1.0 - SMOOTHING_FACTOR) * action_normalized
    )

    # Convert to real-world values
    action_real = denormalize_action(action_smoothed)
    action_real = clip_action(action_real)

    # Send to CoppeliaSim
    sim.callScriptFunction(
        'applyAction',
        script_handle,
        action_real.tolist()
    )

    prev_action = action_smoothed

# --------------------------------------------------
# RUN INFERENCE
# --------------------------------------------------
if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    time.sleep(1)

print("🚀 VLA inference running. Press Ctrl+C to stop.")

try:
    while True:
        start_time = time.time()

        # Get observation
        top_tensor, wrist_tensor, state_tensor = get_observation()

        # Predict action
        with torch.no_grad():
            action = model(
                top_tensor,
                wrist_tensor,
                state_tensor,
                text_tokens=text_tokens
            )
            action = action.cpu().numpy()[0]

        # Apply action
        apply_action(action)

        # Maintain control frequency
        elapsed = time.time() - start_time
        time.sleep(max(0, (1.0 / CONTROL_FREQUENCY) - elapsed))

except KeyboardInterrupt:
    print("\n🛑 Stopping simulation...")

finally:
    sim.stopSimulation()
    print("✅ Simulation stopped safely.")