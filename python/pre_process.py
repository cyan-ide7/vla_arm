# python/pre_process.py
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

dataset_dir = "dataset"
processed_dir = "processed_dataset"
os.makedirs(processed_dir, exist_ok=True)

IMAGE_SIZE = (224, 224)
STATE_DIM = 13
ACTION_DIM = 6


# --------------------------------------------------
# STATE NORMALIZATION
# --------------------------------------------------
def normalize_state(state):
    """
    Ensures state = 13D:
    [joint(6), pos(3), rpy(3), gripper(1)]
    """
    state = np.array(state, dtype=np.float32).flatten()

    if len(state) < STATE_DIM:
        padded = np.zeros(STATE_DIM, dtype=np.float32)
        padded[:len(state)] = state
        return padded

    return state[:STATE_DIM]


# --------------------------------------------------
# ACTION GENERATION (FIXED → 6D ONLY)
# --------------------------------------------------
def generate_actions(states):
    """
    Generate 6D actions:
    [dx, dy, dz, dpitch, dyaw, gripper]
    """
    num_steps = len(states)
    actions = np.zeros((num_steps, ACTION_DIM), dtype=np.float32)

    for i in range(num_steps - 1):
        # -----------------------------
        # Position deltas
        # -----------------------------
        actions[i, 0:3] = states[i + 1, 6:9] - states[i, 6:9]

        # -----------------------------
        # Orientation deltas (FIXED)
        # -----------------------------
        pitch_prev = states[i, 10]
        pitch_next = states[i + 1, 10]

        yaw_prev = states[i, 11]
        yaw_next = states[i + 1, 11]

        # Wrap angles properly (important)
        dpitch = np.arctan2(np.sin(pitch_next - pitch_prev), np.cos(pitch_next - pitch_prev))
        dyaw = np.arctan2(np.sin(yaw_next - yaw_prev), np.cos(yaw_next - yaw_prev))

        actions[i, 3] = dpitch
        actions[i, 4] = dyaw

        # -----------------------------
        # Gripper
        # -----------------------------
        actions[i, 5] = states[i + 1, 12]

    return actions


print("Starting dataset preprocessing...\n")

for traj in os.listdir(dataset_dir):
    traj_path = os.path.join(dataset_dir, traj)

    if not os.path.isdir(traj_path):
        continue

    print(f"Processing trajectory: {traj}")

    top_dir = os.path.join(traj_path, "top_images")
    wrist_dir = os.path.join(traj_path, "wrist_images")
    states_path = os.path.join(traj_path, "states.npy")
    instruction_path = os.path.join(traj_path, "instruction.txt")

    if not os.path.exists(states_path):
        print(f"⚠️ Skipping {traj}: states.npy not found.")
        continue

    # --------------------------------------------------
    # CREATE OUTPUT DIR
    # --------------------------------------------------
    processed_traj_dir = os.path.join(processed_dir, traj)
    os.makedirs(os.path.join(processed_traj_dir, "top_images"), exist_ok=True)
    os.makedirs(os.path.join(processed_traj_dir, "wrist_images"), exist_ok=True)

    # --------------------------------------------------
    # LOAD + NORMALIZE STATES
    # --------------------------------------------------
    raw_states = np.load(states_path, allow_pickle=True)
    states = np.array([normalize_state(s) for s in raw_states], dtype=np.float32)

    # --------------------------------------------------
    # GENERATE ACTIONS (6D)
    # --------------------------------------------------
    actions = generate_actions(states)

    # --------------------------------------------------
    # SAVE STATES + ACTIONS
    # --------------------------------------------------
    np.save(os.path.join(processed_traj_dir, "states.npy"), states)
    np.save(os.path.join(processed_traj_dir, "actions.npy"), actions)

    # --------------------------------------------------
    # IMAGE PROCESSING
    # --------------------------------------------------
    for i in tqdm(range(len(states)), desc=f"Images ({traj})"):
        top_img_path = os.path.join(top_dir, f"{i:06d}.png")
        wrist_img_path = os.path.join(wrist_dir, f"{i:06d}.png")

        if not os.path.exists(top_img_path) or not os.path.exists(wrist_img_path):
            continue

        top_img = cv2.imread(top_img_path)
        wrist_img = cv2.imread(wrist_img_path)

        top_img = cv2.resize(top_img, IMAGE_SIZE).astype(np.float32) / 255.0
        wrist_img = cv2.resize(wrist_img, IMAGE_SIZE).astype(np.float32) / 255.0

        np.save(os.path.join(processed_traj_dir, "top_images", f"{i:06d}.npy"), top_img)
        np.save(os.path.join(processed_traj_dir, "wrist_images", f"{i:06d}.npy"), wrist_img)

    # --------------------------------------------------
    # COPY INSTRUCTION
    # --------------------------------------------------
    if os.path.exists(instruction_path):
        with open(instruction_path, "r") as f:
            instruction = f.read()

        with open(os.path.join(processed_traj_dir, "instruction.txt"), "w") as f:
            f.write(instruction)

    print(f"✅ Finished processing {traj}\n")

print("🎉 Dataset preprocessing completed successfully!")