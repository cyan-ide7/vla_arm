import os
import numpy as np
import cv2
from tqdm import tqdm

dataset_dir = "dataset"
processed_dir = "processed_dataset"
os.makedirs(processed_dir, exist_ok=True)

IMAGE_SIZE = (224, 224)
STATE_DIM = 13
ACTION_DIM = 6


def normalize_state(state):
    """Ensures state = 13D: [joint(6), pos(3), rpy(3), gripper(1)]"""
    state = np.array(state, dtype=np.float32).flatten()
    if len(state) < STATE_DIM:
        padded = np.zeros(STATE_DIM, dtype=np.float32)
        padded[:len(state)] = state
        return padded
    return state[:STATE_DIM]


# --------------------------------------------------
# CHANGED: absolute EE actions instead of deltas
# action[i] = the absolute EE pose the robot should
# be targeting at step i.
# Indices in state:
#   6:9  -> EE position  (x, y, z)
#   10   -> pitch
#   11   -> yaw
#   12   -> gripper target [0, 1]
# --------------------------------------------------
def generate_actions(states):
    """
    Generate 6D ABSOLUTE actions: [x, y, z, pitch, yaw, gripper]
    These match the ACTION_LOW / ACTION_HIGH bounds in action_config.py.
    We drop the last sample to avoid a meaningless final "stay still" label.
    """
    num_steps = len(states) - 1          # drop last frame
    actions = np.zeros((num_steps, ACTION_DIM), dtype=np.float32)

    for i in range(num_steps):
        actions[i, 0:3] = states[i, 6:9]    # absolute x, y, z
        actions[i, 3]   = states[i, 10]     # absolute pitch
        actions[i, 4]   = states[i, 11]     # absolute yaw
        actions[i, 5]   = states[i, 12]     # gripper target [0, 1]

    return actions


# --------------------------------------------------
# NEW: extract the goal pose from the final state
# of the trajectory. This is what Claude will
# approximate at inference time, and what the model
# is conditioned on during training.
# goal_pose = [x, y, z, pitch, yaw, gripper]  (6D)
# --------------------------------------------------
def extract_goal_pose(states):
    """
    The goal pose is the EE pose at the LAST step of the trajectory.
    This represents where the arm ends up after completing the task.
    """
    last = states[-1]
    return np.array([
        last[6], last[7], last[8],   # x, y, z
        last[10],                     # pitch
        last[11],                     # yaw
        last[12]                      # gripper
    ], dtype=np.float32)


print("Starting dataset preprocessing...\n")

for traj in os.listdir(dataset_dir):
    traj_path = os.path.join(dataset_dir, traj)

    if not os.path.isdir(traj_path):
        continue

    print(f"Processing trajectory: {traj}")

    top_dir   = os.path.join(traj_path, "top_images")
    wrist_dir = os.path.join(traj_path, "wrist_images")
    states_path      = os.path.join(traj_path, "states.npy")
    instruction_path = os.path.join(traj_path, "instruction.txt")

    if not os.path.exists(states_path):
        print(f"  Skipping {traj}: states.npy not found.")
        continue

    processed_traj_dir = os.path.join(processed_dir, traj)
    os.makedirs(os.path.join(processed_traj_dir, "top_images"),   exist_ok=True)
    os.makedirs(os.path.join(processed_traj_dir, "wrist_images"), exist_ok=True)

    # Load and normalize states
    raw_states = np.load(states_path, allow_pickle=True)
    states = np.array([normalize_state(s) for s in raw_states], dtype=np.float32)

    # Generate ABSOLUTE actions (now correct)
    actions = generate_actions(states)                 # shape: (N-1, 6)

    # CHANGED: save N-1 states to match actions length
    states_trimmed = states[:-1]                       # drop last state too

    np.save(os.path.join(processed_traj_dir, "states.npy"),  states_trimmed)
    np.save(os.path.join(processed_traj_dir, "actions.npy"), actions)

    # NEW: save the goal pose (final EE pose of this trajectory)
    goal_pose = extract_goal_pose(states)
    np.save(os.path.join(processed_traj_dir, "goal_pose.npy"), goal_pose)
    print(f"  Goal pose: {goal_pose}")

    # Image processing — only up to N-1 frames
    num_samples = len(states_trimmed)
    for i in tqdm(range(num_samples), desc=f"  Images ({traj})"):
        top_img_path   = os.path.join(top_dir,   f"{i:06d}.png")
        wrist_img_path = os.path.join(wrist_dir, f"{i:06d}.png")

        if not os.path.exists(top_img_path) or not os.path.exists(wrist_img_path):
            continue

        top_img   = cv2.imread(top_img_path)
        wrist_img = cv2.imread(wrist_img_path)

        top_img   = cv2.resize(top_img,   IMAGE_SIZE).astype(np.float32) / 255.0
        wrist_img = cv2.resize(wrist_img, IMAGE_SIZE).astype(np.float32) / 255.0

        np.save(os.path.join(processed_traj_dir, "top_images",   f"{i:06d}.npy"), top_img)
        np.save(os.path.join(processed_traj_dir, "wrist_images", f"{i:06d}.npy"), wrist_img)

    # Copy instruction text (still useful for logging/debugging)
    if os.path.exists(instruction_path):
        with open(instruction_path, "r") as f:
            instruction = f.read()
        with open(os.path.join(processed_traj_dir, "instruction.txt"), "w") as f:
            f.write(instruction)

    print(f"  Done: {num_samples} samples, actions shape {actions.shape}\n")

print("Dataset preprocessing completed.")