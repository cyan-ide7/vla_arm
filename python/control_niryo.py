# control.py
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

client = RemoteAPIClient()
sim = client.getObject('sim')

sim.startSimulation()
time.sleep(1)

niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

def apply_action(action):
    """
    action: [dx, dy, dz, dRoll, dPitch, dYaw, gripper]
    """
    sim.callScriptFunction('applyAction', script_handle, action)

try:
    print("Executing example motion...")

    apply_action([0.02, 0.0, 0.0, 0, 0, 0, 0])
    time.sleep(2)

    apply_action([0.0, 0.0, -0.02, 0, 0, 0, 0])
    time.sleep(2)

    apply_action([0, 0, 0, 0, 0, 0, 1])  # Close gripper
    time.sleep(2)

    apply_action([0.0, 0.0, 0.05, 0, 0, 0, 1])
    time.sleep(2)

    apply_action([0.0, -0.05, 0.0, 0, 0, 0, 1])
    time.sleep(2)

    apply_action([0, 0, 0, 0, 0, 0, 0])  # Release
    time.sleep(2)

finally:
    sim.stopSimulation()
    print("Simulation stopped.")