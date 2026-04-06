# perception/detector.py
import os
import time
import numpy as np
import cv2
import base64
import io
from PIL import Image
from fastapi import FastAPI
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import uvicorn

app = FastAPI()

COPPELIA_HOST = os.environ.get("COPPELIA_HOST", "host.docker.internal")
COPPELIA_PORT = int(os.environ.get("COPPELIA_PORT", 23000))

print(f"[PERCEPTION] Connecting to CoppeliaSim at {COPPELIA_HOST}:{COPPELIA_PORT}")
client = RemoteAPIClient(host=COPPELIA_HOST, port=COPPELIA_PORT)
sim    = client.getObject('sim')

# ✅ Retry until simulation is running and camera is found
camera = None
while camera is None:
    try:
        camera = sim.getObject('/camera_vla')
        sim.setObjectInt32Param(camera, 1100, 1)
        print(f"[PERCEPTION] Camera ready — handle {camera}")
    except Exception as e:
        print(f"[PERCEPTION] Waiting for sim to start... ({e})")
        time.sleep(2)


def capture_frame():
    try:
        sim.handleVisionSensor(camera)
        image, resolution = sim.getVisionSensorImg(camera, 0)
        if not image:
            return None, None
        img     = np.frombuffer(image, dtype=np.uint8).reshape(resolution[1], resolution[0], 3)
        img     = cv2.flip(img, 0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img_bgr, resolution
    except Exception as e:
        print(f"[PERCEPTION] Capture error: {e}")
        return None, None


def detect_red(img_bgr):
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0,   70, 50]), np.array([10,  255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([179, 255, 255]))
    mask  = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    best = max(contours, key=cv2.contourArea, default=None)

    if best is None or cv2.contourArea(best) < 25:
        return None

    x, y, w, h = cv2.boundingRect(best)
    return {
        "x"   : x,
        "y"   : y,
        "w"   : w,
        "h"   : h,
        "cx"  : x + w // 2,
        "cy"  : y + h // 2,
        "area": float(cv2.contourArea(best))
    }


@app.get("/capture")
def capture():
    img_bgr, resolution = capture_frame()
    if img_bgr is None:
        return {"error": "No image from camera"}

    detection = detect_red(img_bgr)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buf     = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        "image_b64" : img_b64,
        "resolution": resolution,
        "detection" : detection
    }


@app.get("/object_position")
def object_position():
    try:
        cuboid    = sim.getObject('/Cuboid')
        world_pos = sim.getObjectPosition(cuboid, -1)
        return {
            "x": round(world_pos[0], 4),
            "y": round(world_pos[1], 4),
            "z": round(world_pos[2], 4)
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/health")
def health():
    return {"status": "ok", "camera": camera}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)