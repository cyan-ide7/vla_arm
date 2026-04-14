from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import time

client = RemoteAPIClient()
sim = client.getObject('sim')

sim.startSimulation()
time.sleep(1)

niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

obs = sim.callScriptFunction('getObservation', script_handle, [])

print("EE Position:", obs['ee_position'])
print("Joint Positions:", obs['joint_positions'])
print("Gripper State:", obs['gripper_state'])