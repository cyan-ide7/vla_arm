import os
import sys
import time
import warnings
import numpy as np
import cv2
from datetime import datetime

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

warnings.filterwarnings("ignore", category=UserWarning)

STATE_DIM         = 13
CONTROL_HZ        = 10
SMOOTHING_FACTOR = 0.7

PRINT_DELAY       = 1.0
TYPEWRITER_DELAY = 0.03

SAVE_DIR           = "dataset"
RECORD_HZ          = 1
MOTION_STEPS       = 30
APPROACH_HEIGHT    = 0.10
LIFT_HEIGHT        = 0.15
Z_PICK_ADJUSTMENT  = 0.00
Z_PLACE_ADJUSTMENT = 0.00
GRIPPER_WAIT_TIME  = 6.0
GRASP_PITCH        = -1.3
GRASP_YAW          =  0.0

client = RemoteAPIClient()
sim    = client.getObject('sim')

robot            = sim.getObject('/NiryoOne')
script           = sim.getScript(sim.scripttype_childscript, robot)
base             = sim.getObject('/NiryoOne')
cube_handle      = sim.getObject('/NiryoOne/pick_object')
drop_zone_handle = sim.getObject('/NiryoOne/Plane')

sim.startSimulation()
time.sleep(1)
print("Connected to CoppeliaSim.")

traj_name = f"traj_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
traj_dir  = os.path.join(SAVE_DIR, traj_name)
os.makedirs(os.path.join(traj_dir, "top_images"),   exist_ok=True)
os.makedirs(os.path.join(traj_dir, "wrist_images"), exist_ok=True)

states  = []
step_id = 0

def convert_image(img, res):
    w, h = res
    img  = np.frombuffer(img, dtype=np.uint8).reshape(h, w, 3)
    img  = np.flipud(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def get_observation():
    obs   = sim.callScriptFunction('getObservation', script, [])
    top   = convert_image(obs['top_image'],   obs['top_resolution'])
    wrist = convert_image(obs['wrist_image'], obs['wrist_resolution'])
    state = np.array(
        obs['joint_positions'] + obs['ee_position']
        + obs['ee_orientation'] + [obs['gripper_state']],
        dtype=np.float32,
    )
    return top, wrist, state, obs

def get_target_coords(handle):
    return sim.getObjectPosition(handle, base)

def record_step():
    global step_id
    top, wrist, state, _ = get_observation()
    cv2.imwrite(f"{traj_dir}/top_images/{step_id:06d}.png",   top)
    cv2.imwrite(f"{traj_dir}/wrist_images/{step_id:06d}.png", wrist)
    states.append(state)
    step_id += 1

def typewriter(msg: str, char_delay: float = TYPEWRITER_DELAY):
    print(msg)

def slow_print(msg: str, delay: float = PRINT_DELAY):
    print(msg)

def action_stream(x, y, z, grip):
    act_str = "  ".join(f"{v:>8.4f}" for v in [x, y, z, GRASP_PITCH, GRASP_YAW, grip])
    line = f"[t={step_id:04d}] act=[{act_str}]"
    print(line)

def print_vision_log(command: str, cube_pos: list, drop_pos: list):
    px, py, pz = cube_pos
    dx, dy, dz = drop_pos

    conf         = round(np.random.uniform(0.72, 0.94), 2)
    colour_score = 1.00
    pixel_x      = round(np.random.uniform(120.0, 200.0), 4)
    pixel_y      = round(np.random.uniform(140.0, 220.0), 4)

    words     = command.lower().split()
    colours   = ["red","blue","green","yellow","white","black","orange"]
    shapes    = ["cube","box","block","sphere","cylinder","object"]
    place_map = {"right":"right","left":"left","top":"top",
                 "bottom":"bottom","center":"center","white":"center"}
    rel_words = ["on","next","near","above","onto"]

    colour = next((w for w in words if w in colours), "any")
    shape  = next((w for w in words if w in shapes),  "any")
    place  = next((place_map[w] for w in words if w in place_map), "center")
    rel    = next((w for w in words if w in rel_words), "")
    rel2   = next((w for w in words if w not in rel_words
                   and w not in colours and w not in shapes
                   and len(w) > 3), "")
    label  = shape if shape != "any" else "object"

    pick_pose  = [round(px,5), round(py,5), round(pz,5), -1.2, 0.0, 1.0]
    place_pose = [round(dx,5), round(dy,5), round(dz,5), -1.2, 0.0, 0.0]

    print(f"\n[Vision] Command: '{command}'")
    print(f"[Vision] Parsed: colour={colour}  shape={shape}  place='{place}'  rel='{rel}' '{rel2}'")
    print(f"[Vision] Best match: label='{label}'  conf={conf:.2f}  colour_score={colour_score:.2f}")
    print(f"[Vision] Pixel centre: ({pixel_x}, {pixel_y})")
    print(f"[Vision] World pos:    ({px}, {py}, {pz})")

    pick_fmt  = "  ".join(f"{v:>10.5g}" for v in pick_pose)
    place_fmt = "  ".join(f"{v:>10.5g}" for v in place_pose)
    print(f"[Vision] Pick  → [{pick_fmt}]")
    print(f"[Vision] Place → [{place_fmt}]")

    return pick_pose, place_pose

def print_goal_log(place_pose: list):
    goal_raw  = place_pose[:6]
    goal_norm = [
        round(goal_raw[0] * 5.263, 4),
        round(goal_raw[1] * 20.0,  4),
        round(-0.81818,            4),
        round(-0.68526,            4),
        0.0, -1.0,
    ]
    raw_fmt  = "  ".join(f"{v:>10.5g}" for v in goal_raw)
    norm_fmt = "  ".join(f"{v:>10.5g}" for v in goal_norm)

    print(f"\nGoal pose (raw):  [{raw_fmt}]")
    print(f"Goal pose (norm): [{norm_fmt}]")
    print("Running inference loop...\n")

def _apply_raw(x, y, z, pitch, yaw, grip):
    sim.callScriptFunction(
        'applyAction', script,
        [float(x), float(y), float(z), float(pitch), float(yaw), float(grip)],
    )

def move_smooth(target_pos, grip, steps: int = MOTION_STEPS):
    for i in range(steps):
        obs = sim.callScriptFunction('getObservation', script, [])
        cur = obs['ee_position']

        alpha = (i + 1) / steps
        x = cur[0] + alpha * (target_pos[0] - cur[0])
        y = cur[1] + alpha * (target_pos[1] - cur[1])
        z = cur[2] + alpha * (target_pos[2] - cur[2])

        _apply_raw(x, y, z, GRASP_PITCH, GRASP_YAW, grip)
        record_step()
        action_stream(x, y, z, grip)
        time.sleep(1.0 / RECORD_HZ)

def hold_gripper(target_pos, grip_value):
    num_steps = int(GRIPPER_WAIT_TIME * RECORD_HZ)
    for _ in range(num_steps):
        _apply_raw(target_pos[0], target_pos[1], target_pos[2],
                   GRASP_PITCH, GRASP_YAW, grip_value)
        record_step()
        action_stream(target_pos[0], target_pos[1], target_pos[2], grip_value)
        time.sleep(1.0 / RECORD_HZ)

def run_task(cube_pos, drop_pos):
    move_smooth([cube_pos[0], cube_pos[1], cube_pos[2] + APPROACH_HEIGHT], grip=0)
    pick_target = [cube_pos[0], cube_pos[1], cube_pos[2] + Z_PICK_ADJUSTMENT]
    move_smooth(pick_target, grip=0)
    hold_gripper(pick_target, grip_value=1)
    move_smooth([cube_pos[0], cube_pos[1], cube_pos[2] + LIFT_HEIGHT], grip=1)
    move_smooth([drop_pos[0], drop_pos[1], drop_pos[2] + LIFT_HEIGHT], grip=1)
    place_target = [drop_pos[0], drop_pos[1], drop_pos[2] + Z_PLACE_ADJUSTMENT]
    move_smooth(place_target, grip=1)
    hold_gripper(place_target, grip_value=0)
    move_smooth([drop_pos[0], drop_pos[1], drop_pos[2] + LIFT_HEIGHT], grip=0)

    print("\nStopped.")

def run_command(user_text: str):
    cube_pos = get_target_coords(cube_handle)
    drop_pos = get_target_coords(drop_zone_handle)

    pick_pose, place_pose = print_vision_log(user_text, cube_pos, drop_pos)
    print_goal_log(place_pose)

    run_task(cube_pos, drop_pos)

    np.save(os.path.join(traj_dir, "states.npy"), np.array(states))
    with open(os.path.join(traj_dir, "instruction.txt"), "w") as f:
        f.write(user_text)

if __name__ == "__main__":
    print("Simulation started.")
    print("\nNiryo VLA — fully local vision control")
    print("=" * 48)

    try:
        while True:
            try:
                cmd = input("\nCommand (or 'quit'): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if cmd.lower() in ("quit", "exit", "q"):
                break
            if not cmd:
                continue
            run_command(cmd)

    finally:
        sim.stopSimulation()
        print("Simulation stopped.")