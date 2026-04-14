from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import time

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')

# Start simulation
sim.startSimulation()
time.sleep(1)

# Get script handle
niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

# Function to apply an action
def apply_action(action):
    """
    action: [dx, dy, dz, dRoll, dPitch, dYaw, gripper]
    Units: meters and radians
    """
    sim.callScriptFunction(
        'applyAction',
        script_handle,
        action
    )

# Example sequence of actions
try:
    print("Moving robot...")

    # Move forward
    apply_action([0.02, 0.0, 0.0, 0, 0, 0, 0])
    time.sleep(2)

    # Move down
    apply_action([0.0, 0.0, -0.02, 0, 0, 0, 0])
    time.sleep(2)

    # Close gripper
    apply_action([0, 0, 0, 0, 0, 0, 1])
    time.sleep(2)

    # Lift object
    apply_action([0.0, 0.0, 0.05, 0, 0, 0, 1])
    time.sleep(2)

    # Move to the side
    apply_action([0.0, -0.05, 0.0, 0, 0, 0, 1])
    time.sleep(2)

    # Release object
    apply_action([0, 0, 0, 0, 0, 0, 0])
    time.sleep(2)

finally:
    sim.stopSimulation()
    print("Simulation stopped.")