# vla_inference.py
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import cv2
import time
import torch
import torch.nn as nn

# Simple placeholder VLA model
class DummyVLA(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 7)

    def forward(self, x):
        return self.fc(x)

model = DummyVLA()
model.eval()

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
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    return img

def get_observation():
    obs = sim.callScriptFunction('getObservation', script_handle, [])
    top_img = convert_image(obs['top_image'], obs['top_resolution'])
    wrist_img = convert_image(obs['wrist_image'], obs['wrist_resolution'])
    state = np.array(obs['joint_positions'], dtype=np.float32)
    return state

def apply_action(action):
    sim.callScriptFunction('applyAction', script_handle, action.tolist())

try:
    while True:
        state = get_observation()
        state_tensor = torch.tensor(state).unsqueeze(0)

        action = model(state_tensor).detach().numpy()[0]
        apply_action(action)

        time.sleep(0.1)

finally:
    sim.stopSimulation()
    print("Inference finished.")