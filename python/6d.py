import os
import numpy as np
from action_config import clip_action

DATASET_DIR = "processed_dataset"

print("🔄 Regenerating 6D actions from states...")

for traj in os.listdir(DATASET_DIR):
    traj_path = os.path.join(DATASET_DIR, traj)
    states_path = os.path.join(traj_path, "states.npy")

    if not os.path.exists(states_path):
        continue

    states = np.load(states_path).astype(np.float32)

    # Extract absolute actions: [x, y, z, roll, yaw, gripper]
    actions = np.zeros((len(states), 6), dtype=np.float32)
    actions[:, 0] = states[:, 6]   # X
    actions[:, 1] = states[:, 7]   # Y
    actions[:, 2] = states[:, 8]   # Z
    actions[:, 3] = states[:, 9]   # Roll
    actions[:, 4] = states[:, 11]  # Yaw (skip pitch)
    actions[:, 5] = states[:, 12]  # Gripper

    actions = np.array([clip_action(a) for a in actions])
    np.save(os.path.join(traj_path, "actions.npy"), actions)

    print(f"✅ Regenerated actions for {traj}")

print("🎉 All trajectories updated.")