# python/record_demonstration.py
import os
import json
import time
import numpy as np
import cv2
from datetime import datetime
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from action_config import ACTION_LOW, ACTION_HIGH, clip_action

# --------------------------------------------------
# USER INPUT: TASK INSTRUCTION
# --------------------------------------------------
instruction = input(
    "Enter the task instruction (e.g., 'Pick the red cube and place it on the left'): "
)

# --------------------------------------------------
# CREATE TRAJECTORY DIRECTORY
# --------------------------------------------------
base_dir = "dataset"
os.makedirs(base_dir, exist_ok=True)

traj_name = f"traj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
traj_dir = os.path.join(base_dir, traj_name)
top_dir = os.path.join(traj_dir, "top_images")
wrist_dir = os.path.join(traj_dir, "wrist_images")

os.makedirs(top_dir, exist_ok=True)
os.makedirs(wrist_dir, exist_ok=True)

# Save instruction
with open(os.path.join(traj_dir, "instruction.txt"), "w") as f:
    f.write(instruction)

print(f"📁 Recording trajectory: {traj_name}")

# --------------------------------------------------
# CONNECT TO COPPELIASIM
# --------------------------------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')

niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

# Start simulation if not already running
if sim.getSimulationState() == sim.simulation_stopped:
    sim.startSimulation()
    time.sleep(1)

print("✅ Connected to CoppeliaSim and simulation started.")

# --------------------------------------------------
# IMAGE CONVERSION
# --------------------------------------------------
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
# GET OBSERVATION FROM LUA SCRIPT
# --------------------------------------------------
def get_observation():
    obs = sim.callScriptFunction('getObservation', script_handle, [])

    # Convert images
    top_img = convert_image(obs['top_image'], obs['top_resolution'])
    wrist_img = convert_image(obs['wrist_image'], obs['wrist_resolution'])

    # Construct 13D state vector:
    # [joint_positions(6), ee_position(3), ee_orientation_rpy(3), gripper_state(1)]
    state_vector = np.array(
        obs['joint_positions']
        + obs['ee_position']
        + obs['ee_orientation']
        + [obs['gripper_state']],
        dtype=np.float32,
    )

    return top_img, wrist_img, state_vector, obs

# --------------------------------------------------
# GENERATE 6D ACTIONS FROM STATES
# --------------------------------------------------
def generate_actions_from_states(states):
    """
    Generate absolute 6D actions:
    [x, y, z, roll, yaw, gripper]
    """
    actions = []
    for state in states:
        x = state[6]
        y = state[7]
        z = state[8]
        roll = state[9]
        yaw = state[11]  # Skip pitch (state[10])
        gripper = state[12]

        action = np.array([x, y, z, roll, yaw, gripper], dtype=np.float32)
        action = clip_action(action)  # Ensure within limits
        actions.append(action)

    return np.array(actions, dtype=np.float32)

# --------------------------------------------------
# DATA COLLECTION LOOP
# --------------------------------------------------
states = []
metadata = []

step_id = 0
record_frequency = 5  # Hz
dt = 1.0 / record_frequency

print("\n📌 Instructions:")
print(" - Move the robot using the CoppeliaSim GUI sliders.")
print(" - Press Ctrl+C in this terminal to stop recording.\n")

try:
    while True:
        start_time = time.time()

        top_img, wrist_img, state_vector, obs = get_observation()

        # Save images
        cv2.imwrite(os.path.join(top_dir, f"{step_id:06d}.png"), top_img)
        cv2.imwrite(os.path.join(wrist_dir, f"{step_id:06d}.png"), wrist_img)

        # Store state
        states.append(state_vector)

        # Metadata for readability
        metadata.append({
            "step_id": step_id,
            "joint_positions": obs['joint_positions'],
            "ee_position": obs['ee_position'],
            "ee_orientation_rpy": obs['ee_orientation'],
            "gripper_state": obs['gripper_state'],
            "top_image": f"top_images/{step_id:06d}.png",
            "wrist_image": f"wrist_images/{step_id:06d}.png"
        })

        print(f"📸 Recorded step {step_id}")
        step_id += 1

        # Maintain recording frequency
        elapsed = time.time() - start_time
        time.sleep(max(0, dt - elapsed))

except KeyboardInterrupt:
    print("\n🛑 Recording stopped by user.")

# --------------------------------------------------
# SAVE STATES AND ACTIONS
# --------------------------------------------------
states = np.array(states, dtype=np.float32)
actions = generate_actions_from_states(states)

np.save(os.path.join(traj_dir, "states.npy"), states)
np.save(os.path.join(traj_dir, "actions.npy"), actions)

# --------------------------------------------------
# SAVE METADATA
# --------------------------------------------------
with open(os.path.join(traj_dir, "metadata.json"), "w") as f:
    json.dump({
        "instruction": instruction,
        "num_steps": step_id,
        "state_dimension": 13,
        "action_dimension": 6,
        "description": {
            "state": "[joint_positions(6), ee_position(3), ee_orientation_rpy(3), gripper_state(1)]",
            "action": "[x, y, z, roll, yaw, gripper]"
        },
        "action_limits": {
            "low": ACTION_LOW.tolist(),
            "high": ACTION_HIGH.tolist()
        },
        "steps": metadata
    }, f, indent=4)

# --------------------------------------------------
# STOP SIMULATION
# --------------------------------------------------
sim.stopSimulation()
print(f"\n✅ Dataset successfully saved to: {traj_dir}")