import numpy as np

# --------------------------------------------------
# ACTION SPACE DEFINITION
# --------------------------------------------------
# Action vector: [x, y, z, pitch, yaw, gripper]
# Units:
#   x, y, z -> meters
#   pitch, yaw -> radians
#   gripper -> 0 (open) or 1 (closed)

ACTION_LOW = np.array([
    -0.19,                 # X (meters)
     0.25,                 # Y (meters)
    -0.05,                 # Z (meters)
    -np.pi / 2,            # Pitch (-90 degrees)
    -np.deg2rad(20),       # Yaw (-20 degrees)
     0.0                   # Gripper (open)
], dtype=np.float32)

ACTION_HIGH = np.array([
     0.19,                 # X (meters)
     0.50,                 # Y (meters)
     0.50,                 # Z (meters)
     np.pi / 4,            # Pitch (45 degrees)
     np.deg2rad(20),       # Yaw (20 degrees)
     1.0                   # Gripper (closed)
], dtype=np.float32)

# Dimension of the action vector
ACTION_DIM = len(ACTION_LOW)

# --------------------------------------------------
# NORMALIZATION UTILITIES
# --------------------------------------------------
def normalize_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    denom = ACTION_HIGH - ACTION_LOW
    denom[denom == 0] = 1.0
    return 2.0 * (action - ACTION_LOW) / denom - 1.0


def denormalize_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    return (action + 1.0) / 2.0 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW


def clip_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    return np.clip(action, ACTION_LOW, ACTION_HIGH)


# --------------------------------------------------
# TEST ACTION GENERATION
# --------------------------------------------------
def generate_test_actions():
    center = (ACTION_LOW + ACTION_HIGH) / 2.0

    return {
        "center": center,
        "min_limits": ACTION_LOW.copy(),
        "max_limits": ACTION_HIGH.copy(),
        "open_gripper": np.array([
            center[0], center[1], center[2],
            center[3], center[4], 0.0
        ], dtype=np.float32),
        "close_gripper": np.array([
            center[0], center[1], center[2],
            center[3], center[4], 1.0
        ], dtype=np.float32),
    }


# --------------------------------------------------
# DESCRIPTION (UPDATED)
# --------------------------------------------------
ACTION_DESCRIPTION = {
    "0": "End-effector X position (meters)",
    "1": "End-effector Y position (meters)",
    "2": "End-effector Z position (meters)",
    "3": "End-effector Pitch (radians)",   # ✅ FIXED
    "4": "End-effector Yaw (radians)",
    "5": "Gripper state (0=open, 1=closed)",
}

if __name__ == "__main__":
    print("Action configuration loaded successfully.")
    print("ACTION_DIM:", ACTION_DIM)
    print("ACTION_LOW:", ACTION_LOW)
    print("ACTION_HIGH:", ACTION_HIGH)