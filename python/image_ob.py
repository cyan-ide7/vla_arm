# image_ob.py
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import time

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')

sim.startSimulation()
time.sleep(1)

# Get script handle
niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

def convert_image(img_buffer, resolution):
    width, height = resolution
    img = np.frombuffer(img_buffer, dtype=np.uint8)
    img = img.reshape(height, width, 3)
    img = np.flipud(img)  # Correct orientation
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_observation():
    obs = sim.callScriptFunction('getObservation', script_handle, [])
    top_img = convert_image(obs['top_image'], obs['top_resolution'])
    wrist_img = convert_image(obs['wrist_image'], obs['wrist_resolution'])
    return obs, top_img, wrist_img

try:
    while True:
        obs, top_img, wrist_img = get_observation()

        print("EE Position:", obs['ee_position'])
        print("Joint Positions:", obs['joint_positions'])
        print("Gripper State:", obs['gripper_state'])

        cv2.imshow("Top Camera", top_img)
        cv2.imshow("Wrist Camera", wrist_img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(0.05)

finally:
    sim.stopSimulation()
    cv2.destroyAllWindows()