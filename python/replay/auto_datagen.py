import os
import time
import numpy as np
import cv2
from datetime import datetime
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SAVE_DIR = "dataset"
NUM_TRAJECTORIES = 500
RECORD_HZ = 5
GRIPPER_STEPS = 85 # Exactly 6 seconds of recording during grip/release

APPROACH_HEIGHT = 0.10
LIFT_HEIGHT = 0.15
# REACH FIX: Adjust this if the arm misses the cube
Z_PICK_ADJUSTMENT = -0.01 

GRASP_PITCH = -1.4
GRASP_YAW = 0.0
WORLD = -1 

# --- WORKSPACE LIMITS FROM YOUR LOGS ---
X_MIN, X_MAX = -0.18, 0.10
Y_CENTER = 0.546
Y_MIN, Y_MAX = Y_CENTER - 0.10, Y_CENTER + 0.08

# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------
print("🔗 Connecting to CoppeliaSim...")
client = RemoteAPIClient()
sim = client.getObject('sim')
client.setStepping(True) # Synchronous mode for stability

robot = sim.getObject('/NiryoOne')
script = sim.getScript(sim.scripttype_childscript, robot)
base = sim.getObject('/NiryoOne')

cube_handle = sim.getObject('/NiryoOne/pick_object')
drop_zone_handle = sim.getObject('/NiryoOne/Plane')

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def convert_image(img, res):
    if not img: return None
    w, h = res
    img = np.frombuffer(img, dtype=np.uint8)
    img = img.reshape(h, w, 3)
    img = np.flipud(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def record_step(traj_path, states_list, step_id):
    obs = sim.callScriptFunction('getObservation', script, [])
    top = convert_image(obs['top_image'], obs['top_resolution'])
    wrist = convert_image(obs['wrist_image'], obs['wrist_resolution'])
    
    if top is not None:
        cv2.imwrite(f"{traj_path}/top_images/{step_id:06d}.png", top)
    if wrist is not None:
        cv2.imwrite(f"{traj_path}/wrist_images/{step_id:06d}.png", wrist)
    
    state = np.array(obs['joint_positions'] + obs['ee_position'] + 
                     obs['ee_orientation'] + [obs['gripper_state']], dtype=np.float32)
    states_list.append(state)
    return step_id + 1

def apply_action(x, y, z, pitch, yaw, grip):
    sim.callScriptFunction('applyAction', script, [float(x), float(y), float(z), float(pitch), float(yaw), float(grip)])
    client.step()

def move_smooth(target_pos, grip, traj_path, states_list, step_id, steps=20):
    for i in range(steps):
        obs = sim.callScriptFunction('getObservation', script, [])
        cur = obs['ee_position']
        alpha = (i + 1) / steps
        x = cur[0] + alpha * (target_pos[0] - cur[0])
        y = cur[1] + alpha * (target_pos[1] - cur[1])
        z = cur[2] + alpha * (target_pos[2] - cur[2])
        apply_action(x, y, z, GRASP_PITCH, GRASP_YAW, grip)
        step_id = record_step(traj_path, states_list, step_id)
    return step_id

# --------------------------------------------------
# MASTER LOOP
# --------------------------------------------------
os.makedirs(SAVE_DIR, exist_ok=True)

for t in range(NUM_TRAJECTORIES):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    traj_path = os.path.join(SAVE_DIR, f"traj_{timestamp}")
    os.makedirs(os.path.join(traj_path, "top_images"), exist_ok=True)
    os.makedirs(os.path.join(traj_path, "wrist_images"), exist_ok=True)

    # 1. Reset Sim
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped: time.sleep(0.01)
    
    sim.startSimulation()
    client.step()

    # 2. Randomize Cube based on your terminal limits
    rx = np.random.uniform(X_MIN, X_MAX) 
    ry = np.random.uniform(Y_MIN, Y_MAX)
    sim.setObjectPosition(cube_handle, WORLD, [rx, ry, 0.025])
    
    # Let cube settle
    for _ in range(15): client.step()

    # 3. Get fresh coords relative to robot base
    cube_pos = sim.getObjectPosition(cube_handle, base)
    drop_pos = sim.getObjectPosition(drop_zone_handle, base)
    
    states_list = []
    sid = 0

    try:
        print(f"📦 Traj {t+1}: Cube at X={rx:.3f}, Y={ry:.3f}")

        # A. APPROACH
        sid = move_smooth([cube_pos[0], cube_pos[1], cube_pos[2] + APPROACH_HEIGHT], 0, traj_path, states_list, sid)
        
        # B. PICK
        pick_z = [cube_pos[0], cube_pos[1], cube_pos[2] + Z_PICK_ADJUSTMENT]
        sid = move_smooth(pick_z, 0, traj_path, states_list, sid)
        
        # C. GRIP (6 seconds)
        for _ in range(GRIPPER_STEPS):
            apply_action(pick_z[0], pick_z[1], pick_z[2], GRASP_PITCH, GRASP_YAW, 1)
            sid = record_step(traj_path, states_list, sid)

        # D. LIFT & MOVE
        sid = move_smooth([cube_pos[0], cube_pos[1], cube_pos[2] + LIFT_HEIGHT], 1, traj_path, states_list, sid)
        sid = move_smooth([drop_pos[0], drop_pos[1], drop_pos[2] + LIFT_HEIGHT], 1, traj_path, states_list, sid)
        
        # E. DROP
        place_z = [drop_pos[0], drop_pos[1], drop_pos[2] + Z_PICK_ADJUSTMENT]
        sid = move_smooth(place_z, 1, traj_path, states_list, sid)
        
        # F. RELEASE (6 seconds)
        for _ in range(GRIPPER_STEPS):
            apply_action(place_z[0], place_z[1], place_z[2], GRASP_PITCH, GRASP_YAW, 0)
            sid = record_step(traj_path, states_list, sid)

        # G. RETRACT
        sid = move_smooth([drop_pos[0], drop_pos[1], drop_pos[2] + LIFT_HEIGHT], 0, traj_path, states_list, sid)

        # 4. Save
        np.save(os.path.join(traj_path, "states.npy"), np.array(states_list))
        with open(os.path.join(traj_path, "instruction.txt"), "w") as f:
            f.write("pick up the  red cube and place it on white area")

    except Exception as e:
        print(f"❌ Error: {e}")

sim.stopSimulation()
print(" Success! 500 trajectories generated.")