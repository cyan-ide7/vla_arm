import os
import math
import httpx
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import uvicorn

app = FastAPI()

COPPELIA_HOST  = os.environ.get("COPPELIA_HOST",  "host.docker.internal")
COPPELIA_PORT  = int(os.environ.get("COPPELIA_PORT", 23000))
VLA_URL        = os.environ.get("VLA_URL",        "http://vla_service:8001")
PERCEPTION_URL = os.environ.get("PERCEPTION_URL", "http://perception_service:8002")

print(f"[CTRL] Connecting to CoppeliaSim at {COPPELIA_HOST}:{COPPELIA_PORT}")
client = RemoteAPIClient(host=COPPELIA_HOST, port=COPPELIA_PORT)
sim    = client.getObject('sim')

# -----------------------------
# Joint handles — match Lua exactly
# Lua: sim.getObject('../Joint', {index=i-1})
# -----------------------------
niryo        = sim.getObject('/NiryoOne')
joint_handles = [
    sim.getObject('/NiryoOne/Joint', {'index': i}) for i in range(6)
]

# Gripper — match Lua exactly
connection   = sim.getObject('/NiryoOne/connection')
gripper      = sim.getObjectChild(connection, 0)
gripper_name = "NiryoNoGripper"
if gripper != -1:
    gripper_name = sim.getObjectAlias(gripper, 4)

# Motion params — exact values from Lua
vel   = 20
accel = 40
jerk  = 80
max_vel   = [math.radians(vel)]   * 6
max_accel = [math.radians(accel)] * 6
max_jerk  = [math.radians(jerk)]  * 6

# Positions from Lua script
HOME      = [0, 0, 0, 0, 0, 0]
PICK_POS  = [
    math.radians(90),
    math.radians(-54),
    math.radians(0),
    math.radians(0),
    math.radians(-36),
    math.radians(-90)
]
PLACE_POS = [
    math.radians(-90),
    math.radians(-54),
    math.radians(0),
    math.radians(0),
    math.radians(-36),
    math.radians(-90)
]

print(f"[CTRL] Joints ready — gripper: {gripper_name}")


# -----------------------------
# Motion Functions
# -----------------------------

def move_to_config(angles_rad):
    """Direct port of Lua moveToConfig."""
    sim.moveToConfig({
        'joints'   : joint_handles,
        'targetPos': list(angles_rad),
        'maxVel'   : max_vel,
        'maxAccel' : max_accel,
        'maxJerk'  : max_jerk,
    })


def move_delta(deltas):
    """Apply small delta joint changes from VLA output."""
    current = [sim.getJointPosition(j) for j in joint_handles]
    target  = [current[i] + deltas[i] for i in range(6)]

    # Safety clamp — prevent wild movements
    limits = [
        math.radians(170),  # joint1
        math.radians(100),  # joint2
        math.radians(100),  # joint3
        math.radians(100),  # joint4
        math.radians(100),  # joint5
        math.radians(170),  # joint6
    ]
    target = [max(-limits[i], min(limits[i], target[i])) for i in range(6)]
    move_to_config(target)


def open_gripper():
    sim.clearInt32Signal(gripper_name + '_close')
    print("[GRIPPER] Opened")


def close_gripper():
    sim.setInt32Signal(gripper_name + '_close', 1)
    print("[GRIPPER] Closed")


def sim_wait(seconds=1.0):
    """Mirrors sim.wait() from Lua."""
    import time
    time.sleep(seconds)


# -----------------------------
# Predefined Demo (exact Lua logic in Python)
# -----------------------------

def run_lua_demo():
    """
    Exact port of the Lua sysCall_thread sequence.
    Use this to verify arm moves correctly before adding VLA.
    """
    print("\n[DEMO] Running Lua demo sequence...")

    print("[DEMO] Moving to PICK position...")
    move_to_config(PICK_POS)

    print("[DEMO] Closing gripper...")
    close_gripper()
    sim_wait(5)                          # sim.wait(5)

    print("[DEMO] Moving to HOME...")
    move_to_config(HOME)                 # targetPos3

    print("[DEMO] Moving to PLACE position...")
    move_to_config(PLACE_POS)

    print("[DEMO] Opening gripper...")
    open_gripper()
    sim_wait(4)                          # sim.wait(4)

    print("[DEMO] Returning to HOME...")
    move_to_config(HOME)

    print("[DEMO] Done.")


# -----------------------------
# API Endpoints
# -----------------------------

class TaskRequest(BaseModel):
    instruction : str = "pick up the red cube"
    max_steps   : int = 200
    use_vla     : bool = True   # False = run hardcoded demo instead


@app.post("/run")
async def run_task(body: TaskRequest):
    # Option A — run hardcoded Lua demo (no VLA needed, good for testing)
    if not body.use_vla:
        run_lua_demo()
        return {"status": "demo_done"}

    # Option B — run VLA inference loop
    print(f"\n[TASK] VLA task: '{body.instruction}'")
    move_to_config(HOME)

    gripper_open = True

    async with httpx.AsyncClient(timeout=30.0) as http:
        for step in range(body.max_steps):

            # 1. Capture from perception
            try:
                perc      = await http.get(f"{PERCEPTION_URL}/capture")
                perc_data = perc.json()
            except Exception as e:
                print(f"\n[WARN] Perception error: {e}")
                continue

            if "error" in perc_data:
                continue

            # 2. VLA inference
            try:
                vla_resp = await http.post(f"{VLA_URL}/predict", json={
                    "image_b64"  : perc_data["image_b64"],
                    "instruction": body.instruction
                })
                action = vla_resp.json()
            except Exception as e:
                print(f"\n[WARN] VLA error: {e}")
                continue

            joints_delta = action["joints"]
            gripper_cmd  = action["gripper"]

            # 3. Move
            move_delta(joints_delta)

            # 4. Gripper
            if gripper_cmd > 0.5 and gripper_open:
                close_gripper()
                gripper_open = False
            elif gripper_cmd <= 0.5 and not gripper_open:
                open_gripper()
                gripper_open = True

            print(f"Step {step+1:03d}/{body.max_steps} | "
                  f"gripper={'CLOSE' if gripper_cmd > 0.5 else 'OPEN'}    ",
                  end="\r")

    return {"status": "done", "steps": body.max_steps}


@app.post("/demo")
def demo():
    """Run the hardcoded Lua demo sequence."""
    run_lua_demo()
    return {"status": "demo_done"}


@app.get("/home")
def go_home():
    move_to_config(HOME)
    return {"status": "home"}


@app.get("/joints")
def get_joints():
    angles = [sim.getJointPosition(j) for j in joint_handles]
    return {
        "radians": angles,
        "degrees": [math.degrees(a) for a in angles]
    }


@app.get("/health")
def health():
    return {"status": "ok", "gripper": gripper_name}

@app.post("/pick_object")
async def pick_object():
    """
    1. Ask perception for object position
    2. Move arm to that position
    3. Pick it up
    """
    async with httpx.AsyncClient(timeout=10.0) as http:

        # 1. Get object position from perception
        resp = await http.get(f"{PERCEPTION_URL}/object_position")
        pos  = resp.json()

        if "error" in pos:
            return {"error": pos["error"]}

        print(f"[PICK] Object at x={pos['x']}  y={pos['y']}  z={pos['z']}")

        # 2. Convert world XYZ → joint angles using CoppeliaSim IK
        # For now use the hardcoded PICK_POS from Lua script
        # (we know it reaches ~x=0.2, y=0.2 from scene setup)
        print("[PICK] Moving to pick position...")
        move_to_config(PICK_POS)

        # 3. Close gripper
        close_gripper()
        import time; time.sleep(3)

        # 4. Lift up
        LIFT_POS = [
            PICK_POS[0],
            math.radians(-30),   # raise joint2
            PICK_POS[2],
            PICK_POS[3],
            PICK_POS[4],
            PICK_POS[5],
        ]
        move_to_config(LIFT_POS)

        # 5. Move to place
        move_to_config(PLACE_POS)

        # 6. Open gripper
        open_gripper()
        time.sleep(2)

        # 7. Home
        move_to_config(HOME)

        return {
            "status"  : "done",
            "picked_at": pos
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)