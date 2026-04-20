"""
vla_interface.py  (updated for vision_parser)
=============================================
Only change from the previous version:
  - imports vision_parser instead of gemini_parser
  - passes the live camera frame to parse_goal_pose_from_frame()
  - everything else is identical
"""

import time
import os
import warnings
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from vla_model import NiryoVLA
from action_config import ACTION_DIM, normalize_action, denormalize_action, clip_action
from vision_parser import parse_goal_pose_from_frame   # only changed import

warnings.filterwarnings("ignore", category=UserWarning)

STATE_DIM        = 13
MODEL_PATH       = os.path.join("models", "niryo_vla.pth")
CONTROL_HZ       = 10
SMOOTHING_FACTOR = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = NiryoVLA(state_dim=STATE_DIM, action_dim=ACTION_DIM, device=device)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_vla.py first.")
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Model loaded.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def convert_image(img_buffer, resolution):
    if img_buffer is None or len(img_buffer) == 0:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    width, height = resolution
    img = np.frombuffer(img_buffer, dtype=np.uint8).reshape(height, width, 3)
    img = np.flipud(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


client = RemoteAPIClient()
sim    = client.getObject("sim")
niryo_handle  = sim.getObject("/NiryoOne")
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)
print("Connected to CoppeliaSim.")


def get_observation():
    obs = sim.callScriptFunction("getObservation", script_handle, [])

    top_bgr   = convert_image(obs["top_image"],   obs["top_resolution"])
    wrist_bgr = convert_image(obs["wrist_image"], obs["wrist_resolution"])

    top_t   = transform(Image.fromarray(cv2.cvtColor(top_bgr,   cv2.COLOR_BGR2RGB))) \
                  .unsqueeze(0).to(device)
    wrist_t = transform(Image.fromarray(cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB))) \
                  .unsqueeze(0).to(device)

    state_vec = np.array(
        obs["joint_positions"] + obs["ee_position"]
        + obs["ee_orientation"] + [obs["gripper_state"]],
        dtype=np.float32
    )
    state_t = torch.tensor(state_vec).unsqueeze(0).to(device)

    return top_t, wrist_t, state_t, top_bgr   # also return raw BGR for YOLO


prev_action = np.zeros(ACTION_DIM, dtype=np.float32)


def apply_action(action_norm: np.ndarray):
    global prev_action
    smoothed = SMOOTHING_FACTOR * prev_action + (1.0 - SMOOTHING_FACTOR) * action_norm
    real     = clip_action(denormalize_action(smoothed))
    sim.callScriptFunction("applyAction", script_handle, real.tolist())
    prev_action = smoothed


def run_command(user_text: str, debug_vision: bool = False):
    global prev_action
    prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

    # Grab one frame to run YOLO on
    _, _, _, top_bgr = get_observation()

    # Vision parser: YOLO detects object, returns 6D goal pose
    goal_pose_raw  = parse_goal_pose_from_frame(
        user_text, top_bgr, task="place", debug=debug_vision
    )
    goal_pose_norm = normalize_action(goal_pose_raw)
    goal_tensor    = torch.tensor(goal_pose_norm, dtype=torch.float32) \
                         .unsqueeze(0).to(device)

    print(f"\nGoal pose (raw):  {goal_pose_raw}")
    print(f"Goal pose (norm): {goal_pose_norm}")
    print("Running inference. Ctrl+C to stop.\n")

    try:
        while True:
            t0 = time.time()
            top_t, wrist_t, state_t, _ = get_observation()

            with torch.no_grad():
                action_norm = model(top_t, wrist_t, state_t, goal_tensor)
                action_norm = action_norm.cpu().numpy()[0]

            apply_action(action_norm)
            time.sleep(max(0.0, 1.0 / CONTROL_HZ - (time.time() - t0)))

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    if sim.getSimulationState() == sim.simulation_stopped:
        sim.startSimulation()
        time.sleep(1.0)
        print("Simulation started.")

    print("\nNiryo VLA — fully local vision control")
    print("=" * 48)

    while True:
        try:
            cmd = input("\nCommand (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if not cmd:
            continue
        run_command(cmd, debug_vision=True)

    sim.stopSimulation()
    print("Simulation stopped.")