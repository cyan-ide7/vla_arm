import os
import time
import json
import requests
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

WAIT_STEPS = 120
TRAVEL_STEPS = 100
Z_PICK_HEIGHT = -0.04
Z_DROP_HEIGHT = -0.04
Z_APPROACH = 0.15
Z_LIFT = 0.20
GRASP_PITCH = -1.4
GRASP_YAW = 0.0


def llm_parse_instruction(text):
    url = "http://localhost:11434/api/generate"
    prompt = f"Extract PICK color and DROP color from: '{text}'. Return ONLY JSON: {{\"pick\": \"red\", \"drop\": \"white\"}}"
    try:
        response = requests.post(
            url,
            json={"model": "llama3", "prompt": prompt, "stream": False, "format": "json"},
            timeout=5
        )
        return json.loads(response.json()['response'])
    except Exception:
        return {"pick": "red", "drop": "white"}


def pixel_to_world_sync(u, v):
    wx = -0.15 + (u - 83) * (0.125 - (-0.15)) / (165 - 83)
    wy = 0.306 + (v - 127) * (0.352 - 0.306) / (113 - 127)
    return wx, wy


def scan_table(img):
    detected_objects = []
    h, w = img.shape[:2]

    img[int(h * 0.7):h, :] = 0
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    colors = {
        'red': [
            (np.array([0, 150, 50]), np.array([10, 255, 255])),
            (np.array([160, 150, 50]), np.array([180, 255, 255]))
        ],
        'green': [(np.array([35, 100, 50]), np.array([90, 255, 255]))],
        'white': [(np.array([0, 0, 240]), np.array([180, 25, 255]))],
        'black': [(np.array([0, 0, 0]), np.array([180, 255, 60]))]
    }

    for name, ranges in colors.items():
        mask = np.zeros(hsv.shape[:2], dtype="uint8")

        for r in ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, r[0], r[1]))

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > 20:
                x, y, bw, bh = cv2.boundingRect(c)
                wx, wy = pixel_to_world_sync(int(x + bw / 2), int(y + bh / 2))
                detected_objects.append({"color": name, "coords": [wx, wy]})

    return detected_objects


def apply_action(x, y, z, p, yaw, grip):
    sim.callScriptFunction(
        'applyAction',
        script_handle,
        [float(x), float(y), float(z), float(p), float(yaw), float(grip)]
    )
    client.step()


def move_smooth(target_pos, grip):
    obs = sim.callScriptFunction('getObservation', script_handle, [])
    start = obs['ee_position']

    for i in range(TRAVEL_STEPS):
        alpha = (i + 1) / TRAVEL_STEPS

        cx = start[0] + alpha * (target_pos[0] - start[0])
        cy = start[1] + alpha * (target_pos[1] - start[1])
        cz = start[2] + alpha * (target_pos[2] - start[2])

        apply_action(cx, cy, cz, GRASP_PITCH, GRASP_YAW, grip)


# ---------------- MAIN ---------------- #

client = RemoteAPIClient()
sim = client.getObject('sim')

client.setStepping(True)

robot_handle = sim.getObject('/NiryoOne')
script_handle = sim.getScript(sim.scripttype_childscript, robot_handle)

user_text = input("Enter Command: ")
instr = llm_parse_instruction(user_text)

sim.startSimulation()

for _ in range(20):
    client.step()

obs = sim.callScriptFunction('getObservation', script_handle, [])

img = np.frombuffer(obs['top_image'], dtype=np.uint8).reshape(
    obs['top_resolution'][1],
    obs['top_resolution'][0],
    3
)

img = cv2.cvtColor(np.flipud(img), cv2.COLOR_RGB2BGR)

detected = scan_table(img)

pick_obj = next((o for o in detected if o['color'] == instr['pick']), None)
drop_obj = next((o for o in detected if o['color'] == instr['drop']), None)

if pick_obj and drop_obj:

    p, d = pick_obj['coords'], drop_obj['coords']

    move_smooth([p[0], p[1], Z_APPROACH], 0)

    for _ in range(WAIT_STEPS):
        apply_action(p[0], p[1], Z_APPROACH, GRASP_PITCH, GRASP_YAW, 0)

    move_smooth([p[0], p[1], Z_PICK_HEIGHT], 0)

    for _ in range(WAIT_STEPS):
        apply_action(p[0], p[1], Z_PICK_HEIGHT, GRASP_PITCH, GRASP_YAW, 1)

    move_smooth([p[0], p[1], Z_LIFT], 1)
    move_smooth([d[0], d[1], Z_LIFT], 1)

    move_smooth([d[0], d[1], Z_DROP_HEIGHT], 1)

    for _ in range(WAIT_STEPS):
        apply_action(d[0], d[1], Z_DROP_HEIGHT, GRASP_PITCH, GRASP_YAW, 0)

    move_smooth([d[0], d[1], Z_LIFT], 0)

else:
    print("Targets not found")

sim.stopSimulation()