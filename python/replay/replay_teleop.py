# python/replay_demo.py

import os
import time
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
TRAJ_DIR = "dataset/traj_20260419_182520"  
PLAYBACK_HZ = 5
INTERP_STEPS = 5  # smoothness

# --------------------------------------------------
# CONNECT
# --------------------------------------------------
client = RemoteAPIClient()
sim = client.getObject('sim')

robot = sim.getObject('/NiryoOne')
script = sim.getScript(sim.scripttype_childscript, robot)

sim.startSimulation()
time.sleep(1)

print("Replaying:", TRAJ_DIR)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
states = np.load(os.path.join(TRAJ_DIR, "states.npy"))

# --------------------------------------------------
# APPLY ACTION
# --------------------------------------------------
def apply_action(x, y, z, pitch, yaw, grip):
    sim.callScriptFunction(
        'applyAction',
        script,
        [float(x), float(y), float(z),
         float(pitch), float(yaw), float(grip)]
    )

# --------------------------------------------------
# SMOOTH INTERPOLATION
# --------------------------------------------------
def smooth_move(prev, target):
    for i in range(INTERP_STEPS):
        alpha = (i + 1) / INTERP_STEPS
        interp = prev + alpha * (target - prev)

        apply_action(
            interp[0], interp[1], interp[2],
            interp[3], interp[4], interp[5]
        )

        time.sleep(1.0 / (PLAYBACK_HZ * INTERP_STEPS))

# --------------------------------------------------
# MAIN REPLAY
# --------------------------------------------------
prev_action = None

for i, state in enumerate(states):

    # --------------------------------------
    # STATE FORMAT:
    # [joint(6), pos(3), rpy(3), gripper(1)]
    # --------------------------------------

    x, y, z = state[6:9]

    # CRITICAL FIX
    pitch = state[9]   # roll → pitch control
    yaw = state[11]

    grip = state[12]

    current_action = np.array([x, y, z, pitch, yaw, grip], dtype=np.float32)

    if prev_action is None:
        apply_action(x, y, z, pitch, yaw, grip)
    else:
        smooth_move(prev_action, current_action)

    prev_action = current_action.copy()

    print(f"Step {i}: replaying")

# --------------------------------------------------
# END
# --------------------------------------------------
sim.stopSimulation()
print("Replay finished")