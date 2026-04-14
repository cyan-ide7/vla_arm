# collect_dataset.py
import os
import json
import numpy as np
import cv2
import time
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# Dataset directories
dataset_dir = "dataset"
os.makedirs(f"{dataset_dir}/top_images", exist_ok=True)
os.makedirs(f"{dataset_dir}/wrist_images", exist_ok=True)
os.makedirs(f"{dataset_dir}/states", exist_ok=True)
os.makedirs(f"{dataset_dir}/actions", exist_ok=True)

metadata = []

# Connect to CoppeliaSim
client = RemoteAPIClient()
sim = client.getObject('sim')
sim.startSimulation()
time.sleep(1)

niryo_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, niryo_handle)

def convert_image(img_buffer, resolution):
    width, height = resolution
    img = np.frombuffer(img_buffer, dtype=np.uint8)
    img = img.reshape(height, width, 3)
    img = np.flipud(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def get_observation():
    obs = sim.callScriptFunction('getObservation', script_handle, [])
    top_img = convert_image(obs['top_image'], obs['top_resolution'])
    wrist_img = convert_image(obs['wrist_image'], obs['wrist_resolution'])
    state = {
        "ee_position": obs['ee_position'],
        "ee_orientation": obs['ee_orientation'],
        "joint_positions": obs['joint_positions'],
        "gripper_state": obs['gripper_state']
    }
    return top_img, wrist_img, state

def apply_action(action):
    sim.callScriptFunction('applyAction', script_handle, action)

instruction = "Pick and place the object to the left."

actions = [
    [0.02, 0, 0, 0, 0, 0, 0],
    [0, 0, -0.02, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0.05, 0, 0, 0, 1],
    [0, -0.05, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0]
]

for step_id, action in enumerate(actions):
    top_img, wrist_img, state = get_observation()

    cv2.imwrite(f"{dataset_dir}/top_images/{step_id:06d}.png", top_img)
    cv2.imwrite(f"{dataset_dir}/wrist_images/{step_id:06d}.png", wrist_img)

    np.save(f"{dataset_dir}/states/{step_id:06d}.npy", state)
    np.save(f"{dataset_dir}/actions/{step_id:06d}.npy", np.array(action))

    metadata.append({
        "step_id": step_id,
        "instruction": instruction,
        "top_image": f"top_images/{step_id:06d}.png",
        "wrist_image": f"wrist_images/{step_id:06d}.png",
        "state_file": f"states/{step_id:06d}.npy",
        "action_file": f"actions/{step_id:06d}.npy"
    })

    apply_action(action)
    time.sleep(1)

with open(f"{dataset_dir}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

sim.stopSimulation()
print("Dataset collection completed.")